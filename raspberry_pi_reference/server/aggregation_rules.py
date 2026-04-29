import torch
import copy


def FedAvg(client_models, global_model, weights):

    if weights is None:
        weights={client_id : 1/len(client_models) for client_id in client_models}
        
    
    target_state_dict = global_model.state_dict(keep_vars=True)
    average_model = copy.deepcopy(target_state_dict)
    diff = copy.deepcopy(target_state_dict)
    
    # set the average model to 0
    for key in target_state_dict:
        average_model[key].data.fill_(0.)

    # calculate the averaged model
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for client_id in client_models:
                client_model = client_models[client_id]
                average_model[key].data += weights[client_id]*client_model[key].data

    # update the global model
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            diff[key].data = target_state_dict[key].data - average_model[key].data
            target_state_dict[key].data = average_model[key].data
    
    # return the difference of the 
    return diff

def FedMom(client_models, global_model, weights, v_old, eta, beta):
    if weights is None:
        weights={client_id : 1/len(client_models) for client_id in client_models}
    

    target_state_dict = global_model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0) 
    average_model=copy.deepcopy(w0) 

    # initialize the v_old for first time
    if v_old is None:
        v_old = copy.deepcopy(w0)

    v_new=copy.deepcopy(v_old) #v_t+1

    # ini
    for key in target_state_dict:
        diff[key].data.fill_(0.)
        v_new[key].data.fill_(0.)
        average_model[key].data.fill_(0.)
        
    # calculate the average model
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for client_id in client_models:
                average_model[key].data +=weights[client_id]*client_models[client_id][key].data
    
    # update the global model
    for key in target_state_dict:
          v_new[key].data=w0[key].data - eta*(w0[key].data - average_model[key].data)
          target_state_dict[key].data=v_new[key].data + beta*(v_new[key].data-v_old[key].data)
    
    for key in target_state_dict:
        diff[key].data.copy_(w0[key].data-target_state_dict[key].data)
        
    return diff, v_new
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def FedAdam(
        client_models,
        global_model,
        weights,
        m_old,
        v_old,
        mu,
        beta,
        gamma_g
        ):

    print('Fedadam aggregation','\n mu=',mu, ' \gamma_g=',gamma_g, ' beta=',beta)
    eps=1e-8
    if weights is None:
        weights={client_id : 1/len(client_models) for client_id in client_models}


    target_state_dict = global_model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0) 
    mt=copy.deepcopy(w0)
    vt=copy.deepcopy(w0)

    # initiliazation of v_old, m_old for the first time
    if m_old  is None:
        m_old = copy.deepcopy(w0)
        v_old = copy.deepcopy(w0)
        for key in target_state_dict:
            m_old[key].data.fill_(0.)
            v_old[key].data.fill_(0.)

    
    for key in target_state_dict:
        diff[key].data.fill_(0.)
        mt[key].data.fill_(0.)
        vt[key].data.fill_(0.)
        
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for client_id in client_models:
                state_dict = client_models[client_id]
                diff[key].data+=weights[client_id]*(copy.deepcopy(state_dict[key].data)-w0[key].data)
    
    for key in target_state_dict:
        mt[key].data=mu*m_old[key].data + (1-mu)*diff[key].data
        vt[key].data=beta*v_old[key].data + (1-beta)*diff[key].data.mul(diff[key].data)
        denorm=torch.sqrt(vt[key].data)+eps
        # print('demore,=',denorm)
        target_state_dict[key].data=w0[key].data + gamma_g*torch.div(mt[key].data,denorm)
        
    return mt, vt

def FedMGDA(client_models, global_model, weights, gamma=1):

    target_state_dict = global_model.state_dict(keep_vars=True)
    w0 = copy.deepcopy(target_state_dict)
    diff_model = copy.deepcopy(target_state_dict)
    diff = copy.deepcopy(target_state_dict)
    
    # set the diff model to 0
    for key in target_state_dict:
        diff_model[key].data.fill_(0.)

    # calculate the update difference 
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for client_id in client_models:
                client_model = client_models[client_id]
                diff_model[key].data += weights[client_id]*(w0[key].data - client_model[key].data)

    # update the global model
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data -= gamma*diff_model[key].data
   
    # the updated gradient
    for key in target_state_dict:
        diff[key].data.copy_(w0[key].data-target_state_dict[key].data)
    # return the difference of the 
    return diff


def  FedNAG(client_models, client_optimizers, global_model, weights):

    """
        code for FedNAG
    """
    if weights is None:
        weights={client_id : 1/len(client_models) for client_id in client_models}

    
    avg_buffer=[]
    # copy one of the local optimizer
    for client_id in client_optimizers:
        avg_momentum_buffer = copy.deepcopy(client_optimizers[client_id])
        break
    
    for key in avg_momentum_buffer['state'].values():
        avg_buffer.append(torch.zeros_like(key['momentum_buffer']))
        
    ## aggregate the momentum term
    print('aggregating the momentum term')
    for client_id in client_optimizers:
        state_dict_op=client_optimizers[client_id]
        for i,state in enumerate(state_dict_op['state'].values()):
            avg_buffer[i].add_(state['momentum_buffer'],alpha=weights[client_id])
    print('len avg',len(avg_buffer))  


    # aggregate the model   
    target_state_dict = global_model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    average_model=copy.deepcopy(w0)
    
    # set the diff to 0
    for key in target_state_dict:
        average_model[key].data.fill_(0.)
        
    # get the averaged parameter
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for client_id in client_models:
                state_dict = client_models[client_id]
                average_model[key].data += weights[client_id]*copy.deepcopy(state_dict[key].data)

    # update the global model
    for key in target_state_dict:
        target_state_dict[key].data.copy_(average_model[key].data)
        
    # return the buffer for local optimizer initialization
    return avg_buffer




def FastSlowMo(client_models,client_optimizers, global_model, global_buffer,weights,beta):
    """
    code for <<FastSloWMo: Federated Learning with combined worker and aggregator momenta>>'''

    :beta, momentum coefficient
    :global_buffer: global momentum buffer
    """
    if weights is None:
        weights={client_id : 1/len(client_models) for client_id in client_models}

    

    avg_buffer=[]
    # copy one of the local optimizer
    for client_id in client_optimizers:
        avg_momentum_buffer = copy.deepcopy(client_optimizers[client_id])
        break
    
    for key in avg_momentum_buffer['state'].values():
        avg_buffer.append(torch.zeros_like(key['momentum_buffer']))

    ## aggregate the momentum term
    print('aggregating the momentum term')
    for client_id in client_optimizers:
        state_dict_op=client_optimizers[client_id]
        for i,state in enumerate(state_dict_op['state'].values()):
            avg_buffer[i].add_(state['momentum_buffer'],alpha=weights[client_id])
    print('len avg',len(avg_buffer))  

    # get the state dict of global model 
    target_state_dict = global_model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    y = copy.deepcopy(w0)  #new buffer
    diff=copy.deepcopy(w0)
    
    # initialization for global buffer
    if global_buffer is None:
        global_buffer = copy.deepcopy(w0)



    # set the y to 0
    for key in target_state_dict:
        y[key].data.fill_(0.)
        
    # get y(t)
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for client_id in client_models:
                state_dict = client_models[client_id]
                y[key].data += weights[client_id]*copy.deepcopy(state_dict[key].data)
                
    # get x(t), the global_model
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            diff[key].data = y[key].data + beta*(y[key].data-global_buffer[key].data)
    
    # store y(t)
    global_buffer=copy.deepcopy(y)   

    #  load x(t) to the global model       
    for key in target_state_dict:
        target_state_dict[key].data.copy_(diff[key].data)

        
    return avg_buffer, global_buffer


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def DOMO(client_models,
         global_model,
         global_buffer,
         old_model, 
         weights,
         lr,
         K,
         gamma,
         beta,
         mu):


    """
    code for <<coordinating momenta for cross-silo federated learning>>'''
    :beta, momentum coefficient
    : lr, learning rate
    : K, local iteration
    : global_buffer: global momentum buffer
    :gamma_g: global learning rate
    : beta, global momentum coeffficient
    : old_model, model that is one step ahead
    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :type weights: torch.Tensor

    """

    if weights is None:
        weights={client_id : 1/len(client_models) for client_id in client_models}

    # get the state dict of the global model 
    target_state_dict = global_model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0)


    # initialize old model
    if old_model is None:
        old_model=copy.deepcopy(w0)
        
    # get the d_t
    for key in diff:
        diff[key].data.fill_(0.)
        
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for client_id in client_models:
                state_dict = client_models[client_id]
                diff[key].data += (old_model[key].data - state_dict[key].data)*weights[client_id]/(lr*K)
    
    
    #initialize the global buffer
    if global_buffer is None:
        global_buffer=copy.deepcopy(w0)
        for key in global_buffer:
            global_buffer[key].data.fill_(0.)
    
    # update the global buffer
    for key in global_buffer:
        if target_state_dict[key].data.dtype == torch.float32:
            global_buffer[key].data.mul_(beta).add_(diff[key].data)


        
    #update the global model:
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.add_(global_buffer[key].data, alpha=-gamma*lr*K)
            #premomentum fusion
            old_model[key].data =  target_state_dict[key].data - global_buffer[key].data*lr*mu*K
        
    return  global_buffer, old_model

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def MIME( client_models, client_optimizers, 
        global_model, global_buffer,
        weights, beta):
    """
     code for MIME

    :beta, momentum coefficient
    : global_buffer: global momentum buffer
    """

    if weights is None:
        weights={client_id : 1/len(client_models) for client_id in client_models}

 
    # get averaged gradient   
    gradient_buffer=[]

    # copy one of the local optimizer
    for client_id in client_optimizers:
        sum_momentum_buffer = copy.deepcopy(client_optimizers[client_id])
        break
    
    for idx, key in enumerate(sum_momentum_buffer['state'].values()):
        gradient_buffer.append(torch.zeros_like(key['gradient_buffer']))

    for client_id in client_optimizers:
        state_dict_op=client_optimizers[client_id]
        for idx,state in enumerate(state_dict_op['state'].values()):
            gradient_buffer[idx].add_(copy.deepcopy(state['gradient_buffer']),alpha=weights[client_id])

    # initialize the global model
    if global_buffer is None:
        print('+'*30)
        global_buffer=copy.deepcopy(gradient_buffer)
    
    # update the global model
    for idx,value in enumerate(gradient_buffer):
        global_buffer[idx].mul_(beta).add_(gradient_buffer[idx], alpha=1-beta)
            
    # get the state_dict of global model
    target_state_dict = global_model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0)
        
    for key in target_state_dict:
        diff[key].data.fill_(0.)
        
    #calculate the averaged model
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for client_id in client_models:
                state_dict = client_models[client_id]
                diff[key].data += copy.deepcopy(state_dict[key].data)*weights[client_id]
    
    #  load the averaged model to the global model       
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.copy_(diff[key].data)
        

    return  global_buffer

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def FedMoS(
        client_models,
        global_model,
        global_buffer,
        weights,
        K,
        lr,
        beta,
        ):
    """
    code for FedMoS
     parameter: eta=[1,S/M], S is the number of sampled clients
    :beta, momentum coefficient
    : lr, learning rate
    : K, local iteration
    : global_buffer: global momentum buffer
    """

    if weights is None:
        weights={client_id : 1/len(client_models) for client_id in client_models}
    
    target_state_dict = global_model.state_dict(keep_vars=True)
    w0=copy.deepcopy(target_state_dict)
    diff=copy.deepcopy(w0)

    #initilize the global momentum
    if global_buffer is None:
        global_buffer = copy.deepcopy(w0)
        for key in target_state_dict:
            global_buffer[key].data.fill_(0.)
    
    # calculate the difference between the global and local model
    for key in target_state_dict:
        diff[key].data.fill_(0.)
    
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for client_id in client_models:
                state_dict = client_models[client_id]
                diff[key].data += state_dict[key].data * weights[client_id]

    # update the global momumtuem
    for key in global_buffer:
        if target_state_dict[key].data.dtype == torch.float32:
            global_buffer[key].data.mul_(beta).add_(diff[key].data, alpha=-1.0/(lr*K))
                
    # update the global model    
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.add_(global_buffer[key].data, alpha=-lr*K)
        
    return  global_buffer

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def FedGLOMO(
        client_models, 
        old_client_models,
        global_model,
        old_model,
        global_buffer,
        weights,
        beta,
        ):
    """
    code for fedglomo
    """

    if weights is None:
        weights={client_id : 1/len(client_models) for client_id in client_models}
    
    target_state_dict = global_model.state_dict(keep_vars=True)
    w0_new=copy.deepcopy(target_state_dict)
    
    if old_model is None:
        old_model = copy.deepcopy(w0_new)

    w0_old = copy.deepcopy(old_model)
    diff_new=copy.deepcopy(w0_new)
    diff_old=copy.deepcopy(old_model)



    # calculate the difference between the global and local model for old and new model
    for key in target_state_dict:
        diff_new[key].data.fill_(0.)
        diff_old[key].data.fill_(0.)
    
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            for client_id in client_models:
                state_dict_new = client_models[client_id]
                state_dict_old = old_client_models[client_id]
                diff_new[key].data += (w0_new[key].data - state_dict_new[key].data)*weights[client_id]
                diff_old[key].data += (w0_old[key].data - state_dict_old[key].data)*weights[client_id]
    
    
    # update the global momumtuem
    if global_buffer is None:
        global_buffer = copy.deepcopy(diff_new)
        
    else:
        for key in global_buffer:
            if global_buffer[key].data.dtype == torch.float32:
                global_buffer[key].data.mul_(1-beta).add_(diff_new[key].data, alpha=beta)
                global_buffer[key].data.add_(diff_new[key].data-diff_old[key].data, alpha=1-beta)

    #update the old model
    old_model = copy.deepcopy(target_state_dict)
    
    # update the global model    
    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            target_state_dict[key].data.add_(global_buffer[key].data, alpha=-1)
        

    return  old_model, global_buffer