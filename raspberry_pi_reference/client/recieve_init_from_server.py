
# this py file is used to recieve the initial parameters
import pickle
class MQTT_connector:
    def __init__(self, client_topic,global_model_topic, initial_parameter_topic):

        self.client_topic = client_topic
        self.global_model_topic = global_model_topic
        self.global_model = None
        self.initial_parameter_topic = initial_parameter_topic
        self.initial_parameter =None
        self.timestamp= None


    #subscribe to intial_parameter 
    def on_connect(self,client, user_data, flag, rc):
        print('connect with result code'+str(rc))
        client.subscribe(self.initial_parameter_topic, qos=2)
        client.subscribe(self.global_model_topic,qos=2)
    
    def on_message(self, client, userdata, msg):
        # initial parameter
        if msg.topic == self.initial_parameter_topic:
            initial_parameter_data = pickle.loads(msg.payload)
            self.initial_parameter =initial_parameter_data
            print('self.initial_parameter=',self.initial_parameter)
        if msg.topic == self.global_model_topic:
            model = pickle.loads(msg.payload)
            self.timestamp= model['timestamp']
            self.global_model = model['state_dict']
            print('*'*20)
            print('recieved the initial global model')



