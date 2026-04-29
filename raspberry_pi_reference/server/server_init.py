
from model import *
from optim import *
from client import *
from metrics import *
from data_handle import *
from calculate_weights import *
#from torch.utils.tensorboard import SummaryWriter
import time
import threading
import pickle

import paho.mqtt.client as mqtt
import base64
from aggregation_rules import *

def init_client(args_,target_task_id, root_path, logs_root):

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
    # chose one dataset for training, 
    target_task_id = 1 ################################################ target dataset
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue
            
        if task_id==target_task_id:
            logs_path = os.path.join(logs_root, "task_{}".format(task_id))
            break


    os.makedirs(logs_path, exist_ok=True)
    logger =None # = SummaryWriter(logs_path)

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
        )
    
    return client_





class FLServer:
    def __init__(self,  global_client, broker_address, broker_port, 
                    global_topic, client_topics,args, initial_parameter_topic):
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.global_model_topic = global_topic
        self.global_model = global_client.model
        self.global_client = global_client

        self.client_topics = client_topics
        self.client_models={ client_id: {} for client_id in client_topics}
        self.old_client_models = {client_id: {} for client_id in client_topics}
        self.client_optimizers={client_id:{} for client_id in client_topics}
        self.weights={client_id : 1/len(self.client_models) for client_id in self.client_models}
        self.lock = threading.Lock()  # for synchronization
        self.recieved_clients = set() # to trace all the client with corresponding recieved models
        self.initial_parameter = args
        self.initial_parameter_topic = initial_parameter_topic
        
        self.recieved_local_models_flag = 0 # to determine whether the data are recieved
        self.init_num_count = 0 # to count the number of client recieved the intilization parameter
        self.recieved_init_flag =0

        self.time_step =0 # global_rounds
        self.args = args # the parameter
        self.K = None # local_iterations

        # model parameter
        self.old_model = None # for DOMO, FedGLOMO
        self.diff = None
        self.v_old = None # old global model for fedmom
        self.m_old = None # for fedadam
        self.v_old_adam = None # for adam
        self.buffer = None
        self.global_buffer = None # global_buffer for FastSlowMo, need initilization
        
        # Create an MQTT client for the server
        self.server_client = mqtt.Client("fl_server")



    # subscribe the subjects of all clients
    def on_connect(self, client, userdata, flags, rc):
        print(f"Server connected with result code {rc}")
        for client_id in self.client_topics:
            client.subscribe(self.client_topics[client_id],qos=2)
            client.subscribe('init_feedback_' + client_id,qos=2)
            # recieve the optimizer value
            client.subscribe(self.client_topics[client_id]+'_optimizer',qos=2)

    def on_message(self, client, userdata, msg):
        # Handle received model updates from clients
        print('*'*10,'recieve the message')
        with self.lock:
            if 'init_feedback' in  msg.topic:
                print('init_feedback in the topic')
                if msg.payload.decode() == '1':
                    self.init_num_count +=1
                print('payload=', msg.payload.decode())
                print('***********init_count=',self.init_num_count)
                if self.init_num_count == len(self.client_topics):

                    self.recieved_init_flag = 1
                    print('*'*20)
                    print('all the client recieved the initial parameter')

            # check if the server recived the model
            else:
                client_id = self.get_client_id(msg.topic)
                if client_id is not None:
                    model_data =pickle.loads(msg.payload)
                    # update the model cache of the clients
                    self.update_client_model (client_id, model_data)
                    self.recieved_clients.add(client_id)
                    print('len recieved model',len(self.recieved_clients),'\n')
                    
                    # if all the updates are recieved, 
                    # then check if the local model are updated to aggregate the global model
                    if len(self.recieved_clients) == len(self.client_topics):
                        self.recieved_local_models_flag = 1

                    

    def get_client_id(self, topic):
        for client_id, client_topics in self.client_topics.items():
            if topic == client_topics:
                return client_id
        return None

    def update_client_model(self,client_id,model_data):
        #update the cache of the local models
        self.client_models[client_id] = model_data['state_dict']
        self.client_optimizers[client_id] = model_data['optimizer_state_dict']
        self.K = model_data['K']
        self.old_client_models[client_id] = model_data['old_state_dict']
        print(f"*********recieved model from {client_id}, k={self.K}")
        


    def aggregate_global_model(self):

        # apply personal aggregation rule here:
        # aggregation rules are defined in aggregation_rules.py
        print('='*20,f'method={self.args.method}')

        if self.args.method =='FedAvg':
            self.diff = FedAvg(self.client_models, self.global_model, self.weights)
        
        elif self.args.method =='FedMom':
            eta =1 # parameter for fedmom
            self.diff, self.v_old=FedMom(self.client_models, self.global_model, self.weights, self.v_old, eta, self.args.beta)
        
        elif self.args.method == 'FedAdam':
             self.m_old,self.v_old_adam = FedAdam(self.client_models, self.global_model, self.weights, \
                            self.m_old,self.v_old_adam, self.args.mu, self.args.beta, self.args.gamma)
        
        elif self.args.method == 'FedMGDA':
            self.weights = calculate_weights(self.client_models,self.global_model)
            self.diff = FedMGDA(self.client_models, self.global_model,self.weights,self.args.gamma)
        
        elif self.args.method =='FedNAG':
            self.buffer = FedNAG(self.client_models, self.client_optimizers, self.global_model, self.weights)
        
        elif self.args.method == 'FastSlowMo':
            self.buffer, self.global_buffer = FastSlowMo(self.client_models,  self.client_optimizers,self.global_model, self.global_buffer, self.weights, self.args.beta)
        
        elif self.args.method == 'DOMO':
            self.buffer, self.old_model = DOMO(self.client_models, self.global_model, self.buffer, self.old_model, 
                self.weights, self.args.lr,self.K, self.args.gamma,self.args.beta, self.args.mu)
        
        elif self.args.method == 'MIME':
            self.buffer = MIME( self.client_models, self.client_optimizers, self.global_model, self.buffer,self.weights, self.args.beta)
        
        elif self.args.method == 'FedMoS':
            self.buffer = FedMoS(self.client_models, self.global_model, self.buffer,self.weights,
                                 self.K, self.args.lr, self.args.beta)

        elif self.args.method =='FedGLOMO':
            self.old_model, self.buffer =FedGLOMO(self.client_models, 
                self.old_client_models, self.global_model, self.old_model,
                self.buffer, self.weights, self.args.beta)

        elif self.args.method =='FedAvg-M':
            self.buffer = FedAvg(self.client_models, self.global_model, self.weights)
        
        elif self.args.method =='FedAvg-M-Mom':
            eta=1
            self.buffer, self.v_old=FedMom(self.client_models, self.global_model, self.weights, self.v_old, eta, self.args.mu)
        
        elif self.args.method =='FedMGDA-M':
            self.weights = calculate_weights(self.client_models,self.global_model)
            self.buffer= FedMGDA(self.client_models, self.global_model,self.weights,self.args.gamma)
        
        elif self.args.method =='FedMGDA-Mom':
            self.weights = calculate_weights(self.client_models,self.global_model)
            eta=1
            self.buffer, self.v_old=FedMom(self.client_models, self.global_model, self.weights, self.v_old, eta, self.args.mu)
        
        elif self.args.method =='FedMGDA-M-Mom':
            self.weights = calculate_weights(self.client_models,self.global_model)
            eta=1
            self.buffer, self.v_old=FedMom(self.client_models, self.global_model, self.weights, self.v_old, eta, self.args.mu)
        
        # self.global_model ={key: self.client_models[client_id][key] for client_id, model in self.client_models.items() for key in model}

    def publish_global_model(self):
        timestamp = time.time()
        global_model_payload = pickle.dumps({'timestamp':timestamp, 'state_dict':self.global_model.state_dict(), 
                            'buffer':self.buffer})
        self.server_client.publish(self.global_model_topic, global_model_payload, qos=2)
        print('pubulihed global model to topic: ', self.global_model_topic)

    def publish_initialization_parameter(self):

        initial_parameter_payload = pickle.dumps(vars(self.initial_parameter))
        #initial_parameter_payload = pickle.dumps({'c':3,'d':4})
        self.server_client.publish(self.initial_parameter_topic, initial_parameter_payload, qos=2)
        print(f"published initial parameter to topic: ", self.initial_parameter_topic)
        #print('initial parameter: ', initial_parameter_payload)

    def start(self):
        self.server_client.on_connect = self.on_connect
        self.server_client.on_message = self.on_message
        self.server_client.connect(self.broker_address, self.broker_port, 360)
        self.server_client.loop_start()
        
    # def broadcast_global_model(self):
    #     # Encode and broadcast the global model to all clients
    #     serialized_model = pickle.dumps(self.global_model)
    #     encoded_model = base64.b64encode(serialized_model)
    #     self.server_client.publish(self.topic, payload=encoded_model, qos=0)


