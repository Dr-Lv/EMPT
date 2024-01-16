# -*- coding: utf-8 -*-
import torch
from torch import autograd
from torch.nn import functional as F
from transformers import AdamW,get_linear_schedule_with_warmup
from dataloader import FewshotDataset
import sys
import time

from model import EPTML


def train_one_batch(idx,class_names,support0,support_label,query0,query_label,net,task_lr,device=1):
    '''
    idx:                batch index         
    class_names：       N categories names (or name id)             List[class_name * N]
    support0:           raw support texts                           List[{tokens:[],h:[],t:[]} * (N*K)]
    support_label:      support instance labels                     [N*K]
    query0:             raw query texts                             List[{tokens:[],h:[],t:[]} * total_Q]
    query_label:        query instance labels                       [total_Q]
    net:                EPTML model       
    task_lr:            fast-tuning learning rate for task-adaptation
    '''
    support, query = net.coder(support0), net.coder(query0) # [N*K,bert_size]
    net.W[idx] = net.prework(class_names)
    logits_q = net(support, support_label, query, net.W[idx])

    return net.loss(logits_q, query_label),   net.accuracy(logits_q, query_label)


def test_model(data_loader,model,val_iter,task_lr,device=1):
    accs=0.0
    model.eval()

    start_time = time.time()
    for it in range(val_iter):
        net = model
        class_name,support,support_label,query,query_label = data_loader[0]
        loss,right = train_one_batch(0,class_name, support, support_label,query,query_label,net,task_lr,device=device)
        accs += right 
        sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * accs / (it+1)) + '\r')
        sys.stdout.flush()
    end_time = time.time()
    avg_speed = val_iter / (end_time - start_time)
    #print('avg_infer_speed: {0:2.2f} tasks/seconds'.format(avg_speed))

    return accs/val_iter


def train_model(model:EPTML, B,N,K,Q,data_dir,
            meta_lr=5e-5, 
            task_lr=1e-2,
            weight_decay = 1e-2,
            train_iter=2000,
            val_iter=2000,
            val_step=50,
            save_ckpt = None,
            load_ckpt = None,
            best_acc = 0.0,
            fp16 = True,
            device = 1,
            warmup_step = 200):

    n_way_k_shot = str(N) + '-way-' + str(K) + '-shot'
    print('Start training ' + n_way_k_shot)
    cuda = torch.cuda.is_available()
    if cuda: model = model.cuda(device)

    if load_ckpt:
        state_dict = torch.load(load_ckpt)['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print('ignore {}'.format(name))
                continue
            print('load {} from {}'.format(name, load_ckpt))
            own_state[name].copy_(param)
    
    
    data_loader={}
    data_loader['train'] = FewshotDataset(data_dir['train'],N,K,Q,data_dir['noise_rate'],data_dir['ins_per_class'],device=device) 
    data_loader['val'] = FewshotDataset(data_dir['val'],N,K,Q,data_dir['noise_rate'],-1,device=device)
    data_loader['test'] = FewshotDataset(data_dir['test'],N,K,Q,data_dir['noise_rate'],-1,device=device)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    coder_named_params = list(model.coder.named_parameters())
    classifier_named_params = list(model.classifier.named_parameters())

    for name, param in coder_named_params:
        #print(name, param.requires_grad)
        if name in {'bert_ebd.word_embeddings.weight','bert_ebd.position_embeddings.weight','bert_ebd.token_type_embeddings.weight'}:
            param.requires_grad = False
            pass


    optim_params=[{'params':[p for n, p in coder_named_params 
                    if not any(nd in n for nd in no_decay)],'lr':meta_lr,'weight_decay': weight_decay},
                  {'params':[p for n, p in coder_named_params 
                    if any(nd in n for nd in no_decay)],'lr':meta_lr, 'weight_decay': 0.0},
                  {'params':[p for n, p in classifier_named_params],'lr':task_lr}
                ]
       

    meta_optimizer=AdamW(optim_params)
    scheduler = get_linear_schedule_with_warmup(meta_optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter)

    if fp16:
        from apex import amp
        model, meta_optimizer = amp.initialize(model, meta_optimizer, opt_level='O1')

    iter_loss, iter_right, iter_sample = 0.0, 0.0, 0.0

    model.train()

    start_time = time.time()
    for it in range(train_iter):
        meta_loss, meta_right = 0.0, 0.0

        for batch in range(B):
            class_name, support, support_label, query, query_label = data_loader['train'][0]
            loss, right =train_one_batch(batch,class_name,support,support_label,query,query_label,model,task_lr,device=device)
            
            meta_loss += loss
            meta_right += right
        
        meta_loss /= B
        meta_right /= B

        meta_optimizer.zero_grad()
        if fp16:
            with amp.scale_loss(meta_loss, meta_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            meta_loss.backward()
        meta_optimizer.step()
        scheduler.step()
    
        iter_loss += meta_loss
        iter_right += meta_right
        iter_sample += 1 

        sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
        sys.stdout.flush()

        if (it+1)%val_step ==0 or it == 0:
            print("")
            avg_speed = val_step*B /(time.time() - start_time)
            #print('avg_train_speed: {0:2.2f} tasks/seconds'.format(avg_speed))

            iter_loss, iter_right, iter_sample = 0.0,0.0,0.0
            acc = test_model(data_loader['val'], model, val_iter, task_lr,device=device)
            print("")
            model.train()
            if acc > best_acc or it+1 == 300:
                print('Best checkpoint!')
                torch.save({'state_dict': model.state_dict()}, save_ckpt)

                best_acc = acc

            start_time = time.time()

    print("\n####################\n")
    print('Finish training model! Best acc: '+str(best_acc))


def eval_model(model,N,K,Q,eval_iter=10000,task_lr=1e-2, noise_rate = 0,file_name=None,load_ckpt = None, device=1):
    if torch.cuda.is_available(): model = model.cuda(device)
    acc_list = []
    if load_ckpt:
        state_dict = torch.load(load_ckpt)['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                # print('ignore {}'.format(name))
                continue
            # print('load {} from {}'.format(name, load_ckpt))
            own_state[name].copy_(param)

    accs=0.0
    model.eval()
    data_loader = FewshotDataset(file_name,N,K,Q,noise_rate,-1,device=device)
    tot = {}
    neg = {}
    for it in range(eval_iter):
        net = model
        class_name,support,support_label,query,query_label = data_loader[0]
        _,right = train_one_batch(0,class_name, support, support_label,query,query_label,net,task_lr,device=device)
        accs += right 
        for i in class_name:
            if i not in tot:
                tot[i]=1
            else:
                tot[i]+=1
        if right <1:
            for i in class_name:
                if i not in neg:
                    neg[i]=1
                else:
                    neg[i]+=1
        sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * accs / (it+1)) + '\r')
        sys.stdout.flush()
        acc_list.append([it, 100 * accs / (it+1)])
    print("")
    print(tot)
    print(neg)
    print("")
    sav_list(acc_list, file_name, K)

    return accs/eval_iter

def sav_list(alist, file_name, K):
    benchmark = file_name.split('/')[2]
    acc = '{0:3.2f}'.format(alist[-1][1])
    fname = benchmark +'_'+ str(K)+'_'+ acc
    with open(fname, 'a') as f:
        for it, acc in alist:
            f.write('{0:5d}, {1:3.2f},\n'.format(it, acc))
 
