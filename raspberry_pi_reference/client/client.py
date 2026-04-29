
# first of all, recieve the initilization parameter from the server


import paho.mqtt.client as mqtt
import base64
import pickle
import time
import argparse
from recieve_init_from_server import *
import os

from model import *
from optim import *
from client_init import *
from metrics import *
from data_handle import *
from torch.utils.tensorboard import SummaryWriter
import time




def args_to_string(args):
    """
    Transform experiment's arguments into a string
    :param args:
    :return: string
    """
    args_string = ""
    args_to_show = ["experiment", "method"]
    for arg in args_to_show:
        args_string = os.path.join(args_string, str(getattr(args, arg)))
    if args.locally_tune_clients:
        args_string += "_adapt"

    return args_string

###### client initilization *******************************************
#**********************************************************************
def init_client(args_,target_task_id, root_path, logs_root,
                client_id, broker_address, broker_port,global_topic, client_topic):

    # get the data
    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators =\
        get_loaders(  
            type_=args_.experiment,
            root_path=root_path,
            batch_size=args_.bz,
            is_validation=args_.validation,
            num_workers=args_.num_workers
        )
        
    for train_iterator in train_iterators:
        n_train_samples=len(train_iterator.dataset)
        print('n_train_samples',n_train_samples)

    print("===> Initializing clients..")
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue
            
        if task_id==target_task_id:
            logs_path = os.path.join(logs_root, "task_{}".format(task_id))
            break


    os.makedirs(logs_path, exist_ok=True)
    logger = SummaryWriter(logs_path)

    n_train_samples=len(train_iterator.dataset)
    local_iteration=args_.local_steps*np.ceil(n_train_samples/args_.bz)
    # get the model
    torch.manual_seed(args_.seed)
    model = FemnistCNN(num_classes=62).to(args_.device)
    criterion = nn.CrossEntropyLoss(reduction="none").to(args_.device)
    metric = accuracy

    # get model
    optimizer=get_optimizer(optimizer_name=args_.optimizer, 
                            model=model,
                            lr_initial=args_.lr,
                            mu=args_.mu,
                            beta=args_.beta,
                            K=local_iteration)

    # initialize the client
    client_=Client(
        model=model,
        criterion=criterion,
        metric=accuracy,
        device=args_.device,
        optimizer=optimizer,
        train_iterator=train_iterator, 
        val_iterator=val_iterator,
        test_iterator=test_iterator,
        logger=logger,
        local_steps=args_.local_steps,
        local_iterations = local_iteration,
        client_id = client_id,
        broker_address = broker_address,
        broker_port = broker_port,
        global_topic = global_topic,
        client_topic = client_topic
        )
    
    return client_


#**************************************************************************
#***********   recieving model and initial parameter **********************
client_topic = "FL_client1"
client_id='client1'
global_model_topic ='global_model'
initial_parameter_topic ='initial_parameter'
# MQTT broker for the federated training cluster.
# Set broker_address to the IP / hostname of the machine running the MQTT
# broker (e.g., the same host running server.py with mosquitto installed,
# or a separate broker instance accessible from every Pi client).
broker_address = "localhost"
broker_port = 1883

# initialization
# set the client to recieve args and model from the server
mqtt_init = MQTT_connector(client_topic,global_model_topic, initial_parameter_topic)
client = mqtt.Client()

client.on_connect = mqtt_init.on_connect
client.on_message = mqtt_init.on_message

client.connect(broker_address, broker_port, 60)
client.loop_start()
# model_data = pickle.dumps({'a':1,'b':2})
# print(model_data)

args=None

# recieve the initial_parameter from server
while args is None or mqtt_init.global_model is None:
    args = mqtt_init.initial_parameter
    time.sleep(2)
    init_model_state_dict = mqtt_init.global_model
    init_timestamp = mqtt_init.timestamp

# give feedback to server after recieved the initial parameters
client.publish('init_feedback_client1',1,qos=2)
time.sleep(5)
client.loop_stop()

# change the dict to namespace
args = argparse.Namespace(**args)
print('args=',args)

# set the root path for data and logs path for results
data_dir = os.path.join("data", args.experiment, "all_data") # in utils  
root_path =  os.path.join(data_dir, "train")

logs_root = os.path.join("logs", args_to_string(args))
logs_root += '_R'+ str(args.n_rounds)+ '_lr'+str(args.lr) +'_Bz'+str(args.bz)+ \
    '_LocalE'+str(args.local_steps) + '_gamma'+str(args.gamma) +'_beta'+str(args.beta) + '_mu'+str(args.mu)


# initialize the client
Fed_client = init_client(args, target_task_id=1,
                    root_path = os.path.join(data_dir, "train"),
                    logs_root = os.path.join(logs_root, "train"),
                    client_id = client_id,
                    broker_address = broker_address, 
                    broker_port = broker_port,
                    global_topic =global_model_topic,
                    client_topic =client_topic
                    )

# load the initial model
Fed_client.model.load_state_dict(init_model_state_dict)
Fed_client.timestamp.append(init_timestamp)

################################################################################
#******************  start training *******************************************
# start training
print("Training..")
pbar = tqdm(total=args.n_rounds)
current_round = 0

test_acc_all, test_loss_all, train_acc_all, train_loss_all =[],[],[],[]

start=time.time()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#  main 
# connect to mqtt
Fed_client.start()

while current_round <=int(args.n_rounds):
    
    # record the result
    if current_round%args.log_freq == 0:
        train_loss, train_acc, test_loss, test_acc =Fed_client.write_logs()
        print(f'train loss={train_loss: .4f},  train acc ={train_acc: .4f} ')
        print(f'test loss={test_loss: .4f},  test acc ={test_acc: .4f} \n')
        train_acc_all.append(train_acc)
        train_loss_all.append(train_loss)
        test_acc_all.append(test_acc)
        test_loss_all.append(test_loss)

    if args.method =='MIME': # get the full gradient
        # compute the full gradient
        Fed_client.compute_full_gradients(Fed_client.train_iterator)
        # set the gradient
        Fed_client.optimizer.set_full_gradient_MIME()
    
    if args.method == 'FedMoS':
        # compute the full gradient
        Fed_client.compute_full_gradients(Fed_client.train_iterator)
        # set the gradient
        Fed_client.optimizer.set_initial_buffer_FedMoS()

    if args.method == 'FedGLOMO':
        if Fed_client.old_global_model is None:
            Fed_client.old_global_model = copy.deepcopy(init_model_state_dict)
        Fed_client.model.load_state_dict(Fed_client.old_global_model)
        Fed_client.step()
        old_state_dict =  Fed_client.model.state_dict()

        if  Fed_client.global_model is None:
            Fed_client.global_model = copy.deepcopy(init_model_state_dict)
        Fed_client.model.load_state_dict(Fed_client.global_model)

        # set the current global model as the old_global model
        Fed_client.old_global_model = copy.deepcopy(Fed_client.global_model)

    # update the local model
    Fed_client.step()

    # publish the model to server
    local_model_state_dict = Fed_client.model.state_dict()
    local_optimizer_state_dict = Fed_client.optimizer.state_dict()

    print(f'==========method={args.method}')
    if args.method == 'FedNAG' or args.method =='FastSlowMo':
        Fed_client.send_local_model(local_model_state_dict,local_optimizer_state_dict=local_optimizer_state_dict)
    elif args.method == 'MIME':
         Fed_client.send_local_model(local_model_state_dict,local_optimizer_state_dict=local_optimizer_state_dict)
    elif args.method == 'FedGLOMO':
         Fed_client.send_local_model(local_model_state_dict,
         local_optimizer_state_dict=local_optimizer_state_dict,
         old_state_dict=old_state_dict)
    else: # FedAvg, FedAvg-M, DOMO, FedMGDA, FedMGDA-M, FedMoS
        Fed_client.send_local_model(local_model_state_dict)
    print('send the model')

    # waite to recieve the global model
    while Fed_client.recieved_model_flag == 0:
        
        if current_round == args.n_rounds:
            break
    # set the flag to 0 for next round waiting
    Fed_client.recieved_model_flag=0

    # load the global model as the local model
    Fed_client.update_local_model()

    print('*'*20, '\n the client recieved the global model')

    current_round +=1
    pbar.update(1)


np.save('train_acc.npy',train_acc)
np.save('test_acc.npy',test_acc)

np.save('train_loss.npy',train_loss)
np.save('test_loss.npy',test_loss)


