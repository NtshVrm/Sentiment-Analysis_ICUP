import classifier as cl 
import visualisation as vis 

def stratify():
    cl.final()

def visualisation():
    for i in dir(vis):
        item = getattr(vis,i)
        if callable(item):
            item()

choice = -1     
print('Enter 1 for visualisations and 2 for classifying tweets!')
print('Press 0 to exit.')
choice = (input())

while choice!='1' and choice!='2':
    if choice=='0':
        break
    else:
        print('Invalid choice.')
        print('Enter 1 for visualisations and 2 for classifying tweets!')
        print('Press 0 to exit.')
        choice = (input())
if choice=='1':
    visualisation()

if choice=='2':
    stratify()



    
