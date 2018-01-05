import socket
sock =socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1',9797))
sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) #在客户端开启心跳维护
"""
sock.send('auto')
res = sock.recv(1024)
print res
"""
path = r'E:\Repositories\Emotion-Analyse\Long-Reviews-Emotion-Analyse\test\neg\0_2.txt'
with open(path,'rb') as reader:
    data = reader.read()
    sock.send(data)
    res = sock.recv(1024)
    print res


