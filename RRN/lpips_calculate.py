f = open("reds_lpips.txt", "r")
total=0
count=0
for x in f:
    count+=1
    x=x.split(': ')
    total+=float(x[1])
print(total/count)
    
    