
from min_norm_solvers import *
import copy
from torch.autograd import Variable

def calculate_weights(client_models,global_model):

    # set the initial state dict
    initial_state_dict = global_model.state_dict(keep_vars=True)

    # copy the clients' models/state_dict
    client_models_copy = copy.deepcopy(client_models)

    for client_id in client_models:
        for key in client_models[client_id]:
            # w^i(t+1)-w(t) 
            client_models_copy[client_id][key].data -= copy.deepcopy(initial_state_dict[key].data)
    

    pn={client_id:[] for client_id in client_models}
    param_differ={client_id:[] for client_id in client_models}# w-w^i(t+1)
    for client_id in client_models_copy:
        for param_name, param_value in client_models_copy[client_id].items():
            if param_value is not None:
                param_differ[client_id].append(Variable(param_value.clone(), requires_grad=False))

            
        pn[client_id] = parameter_normalizers(param_differ[client_id], 'l2') #normalization
        pn[client_id]=1
        #print('client_id',client_id,'pn_diff=',pn[client_id] )
        for pr_i in range(len(param_differ[client_id])):
            param_differ[client_id][pr_i]=param_differ[client_id][pr_i]/pn[client_id]

    print('start alpha calculation')
    '''calculate weight'''
    weights, min_norm = MinNormSolver.find_min_norm_element([param_differ[t] for t in client_models])

    weights_dict={client_id:weight for client_id, weight in zip(client_models,weights) }
    print('*******weights=',weights_dict)
    return weights_dict