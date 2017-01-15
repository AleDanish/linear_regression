import tornado.ioloop
import tornado.web
import _thread
import regression
import queue
from math import hypot

FEATURES_NUM = 5
MULTIPLICAND = 10
MEC_SERVERS_MAP = [(10, 5), (20,15)] #(lat, lon) each MEC server available

class MainHandler(tornado.web.RequestHandler):
    def post(self):
        next_values = {}
        users = []
        files = self.request.files
        for file_name in files:
            updateData(file_name, files[file_name][0]['body'].decode("utf-8"))

            #execute regression
            predict_value = regression.main('data/'+file_name)/MULTIPLICAND
            next_values[file_name] = predict_value
            users.append(file_name.split('_')[0])
        calculateNextMove(next_values, users)

# to send data back to the client
#        self.write(next_values)
#        self.finish()

#TO TEST: curl -F file1=@temperature_data -F file2=@temperature_data.scale http://localhost:8888 file2=@temperature_data2

def calculateNextMove(next_values, users):
    print('Calculating the next move...')
    map_user_server = {}
    for user in list(set(users)):
        lat = next_values[user + '_lat']
        lon = next_values[user + '_lon']
        min_dist = 0
        #calculate the distance from each MEC server
        for server in MEC_SERVERS_MAP:
            dist = hypot(lat - server[0], lon - server[1])
            if (min_dist == 0) or (min_dist > dist):
                min_dist = dist
                map_user_server[user]=server
    print('Here is the map users-best servers:', map_user_server)

def updateData(file_name, content):
    with open('data/'+file_name, 'a+') as f:
        values = content.split()
        for i in range(len(values)):
            values[i] = float(values[i]) * MULTIPLICAND
        print(values)
        try:
            while len(values) > 0:
                line = ''
                for i in range(FEATURES_NUM):
                    val = values[i]
                    line += ' ' + str(i+1) + ':' + str(val)
                line = str(values[i+1]) + line + '\n'
                f.write(line)
                values.remove(values[0])
        except IndexError:
            print('Finished to read values')
        f.close

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("WebServer listening on port 8888")
    tornado.ioloop.IOLoop.current().start()
