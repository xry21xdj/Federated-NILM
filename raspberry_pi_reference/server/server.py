import paho.mqtt.client as mqtt
import base64
import pickle
import json
import time
from args import *
from server_init import *



# root to save the results
args = parse_args()
data_dir = os.path.join("data", args.experiment, "all_data") # in utils  
root_path =  os.path.join(data_dir, "train")

logs_root = os.path.join("global_logs", args_to_string(args))
logs_root += '_R'+ str(args.n_rounds)+ '_lr'+str(args.lr) +'_Bz'+str(args.bz)+ \
    '_LocalE'+str(args.local_steps) + '_gamma'+str(args.gamma) +'_beta'+str(args.beta) + '_mu'+str(args.mu)


# avoiding the mismatch between the method and optimizaer
if args.method=='FedMom':
    print('FedMom')
    assert args.optimizer =='sgd','please change the optimizer to sgd for fedmom'
elif args.method=="FedNAG":
    print('FedNAG')
    assert args.optimizer=='nag_sgd', 'please change the optimizer to nag_sgd'
elif args.method=="FastSlowMo":
    print('FastSlowMo')
    assert args.optimizer=='fastslowmo_sgd', 'please change the optimizer to fastslowmo_sgd'
elif args.method=="DOMO":
    print('DOMO')
    assert args.optimizer=='domo_sgd', 'please change the optimizer to domo_sgd'
elif args.method=="MIME":
    print('MIME')
    assert args.optimizer=='mime_sgd', 'please change the optimizer to mime_sgd'
elif args.method=="FedMoS":
    print('FedMoS')
    assert args.optimizer=='fedmos_sgd', 'please change the optimizer to fedmos_sgd'
elif args.method=="FedGLOMO":
    print('FedGLOMO')
    assert args.optimizer=='fedglomo_sgd', 'please change the optimizer to fedglomo_sgd'
elif args.method=="FedAvg-M-Mom":
    print('FedAvg-M-Mom')
    assert args.optimizer=='m_sgd', 'please change the optimizer to m_sgd'
elif args.method=="FedProx":
    print('FedProx')
    assert args.optimizer=='prox_sgd', 'please change the optimizer to prox_sgd'
elif args.method=="FedCM":
    print('FedCM')
    assert args.optimizer=='cm_sgd', 'please change the optimizer to cm_sgd'
elif args.method=="FedAdam":
    print('FedAdam')
    assert args.optimizer=='sgd', 'please change the optimizer to sgd'
elif args.method=="FedMGDA-M":
    print('FedMGDA-M')
    assert args.optimizer=='m_sgd', 'please change the optimizer to m_sgd'
elif args.method=="FedAvg-M":
    print('FedAvg-M')
    assert args.optimizer=='m_sgd', 'please change the optimizer to m_sgd'
elif args.method=="FedMGDA-M-Mom":
    print('FedMGDA-M-Mom')
    assert args.optimizer=='m_sgd', 'please change the optimizer to m_sgd'
elif args.method=="FedAvg-M-Mom":
    print('FedAvg-M-Mom')
    assert args.optimizer=='m_sgd', 'please change the optimizer to m_sgd'
elif args.method=="FedMGDA-Mom":
    print('FedMGDA-Mom')
    assert args.optimizer=='sgd', 'please change the optimizer to sgd'

# initialize the global model
target_task_id=10 # this target is chosen as the one for global validation
global_client = init_client(args,target_task_id, root_path, logs_root)

# recorde the performance of the global model
global_client.write_logs()


# Configuration for the server.
# Set broker_address to the IP / hostname of the MQTT broker; "localhost"
# works when the broker (e.g., mosquitto) runs on the same machine as
# server.py, otherwise use the broker host's IP visible to every client.
broker_address = "localhost"
broker_port = 1883
global_topic = "global_model"
initial_parameter_topic = 'initial_parameter'
client_topics = {'client1':'FL_client1'}
# client_topics = {'client1':'FL_client1',
#                     'client4':'FL_client4'}
# client_topics ={'client1':'FL_client1',
#                 'client2':'FL_client2',
#                 'client3':'FL_client3',
#                 'client4':'FL_client4'}

# Create the FLServer
fl_server = FLServer( global_client, broker_address, broker_port,
                     global_topic, client_topics, args, initial_parameter_topic)



# Start the MQTT client loop for the server
fl_server.start()
timestamp = time.time()
# keep publish the data while the client has not recieved the intial parameters
while fl_server.recieved_init_flag ==0:


    #send the initial parameter and model
    fl_server.publish_initialization_parameter()

    state_dict = {'timestamp': timestamp,
                    'state_dict':fl_server.global_model.state_dict()}
    model_state_byte = pickle.dumps(state_dict)

    #publish the model state_dict
    fl_server.server_client.publish(global_topic, model_state_byte, qos=2)
    time.sleep(10)
    

# start training
print("Training..")
pbar = tqdm(total=args.n_rounds)
current_round = 0

test_acc_all, test_loss_all, train_acc_all, train_loss_all =[],[],[],[]

start=time.time()



while current_round < int(args.n_rounds):

    # to determine whether the message is recieved
    while fl_server.recieved_local_models_flag ==0:
        flag = fl_server.recieved_local_models_flag
    print('++++++ the clients have sent the models at round: ', current_round)
    
    # aggregate the global model
    fl_server.aggregate_global_model()
    print('\n','#'*20)
    print(' the server have aggregated the global model')


    # pubish the global model to the predifined topic
    fl_server.publish_global_model()
    print('\n','*'*20)
    print(' the server have published the global model')
    
    #clear the received clients set for next round sychronization
    fl_server.recieved_clients.clear()

    # update the global_rounds
    fl_server.time_step +=1
    current_round +=1

    # reset the flag
    fl_server.recieved_local_models_flag=0

    # record the result of the global_model
    print(f'recording result of round={current_round}')
    train_loss, train_acc, test_loss, test_acc = fl_server.global_client.write_logs()
    print(f'train loss={train_loss: .4f},  train acc ={train_acc: .4f} ')
    print(f'test loss={test_loss: .4f},  test acc ={test_acc: .4f} \n')
    train_acc_all.append(train_acc)
    train_loss_all.append(train_loss)
    test_acc_all.append(test_acc)
    test_loss_all.append(test_loss)
                        
    time.sleep(1)

    pbar.update(1)

time.sleep(5)
print('finished training======')

np.save('train_acc.npy',train_acc)
np.save('test_acc.npy',test_acc)

np.save('train_loss.npy',train_loss)
np.save('test_loss.npy',test_loss)





# Stop the MQTT client loop for the server when done
fl_server.server_client.loop_stop()
