a="     +123"
b=[]
for i in range(len(a)):
    if (43 == ord(a[i]) or 45 == ord(a[i])) and len(b)>0:
        break
    elif ord(a[i])==45 or ord(a[i])==43:
        if len(b) == 0:
            b.append(a[i])
        else:
            break
    elif ord(a[i])>=48 and ord(a[i])<=57:
        b.append(a[i])
    elif ord(a[i])==32:
        if len(b)==0:
            pass
        else:
            break
    else:
        break
if len(b)!=0:
    if len(b)==1 and (ord(b[0])==43 or ord(b[0])==45):
        print(0)
    else:
        a=int("".join(b))
        print( a if a<2147483647 and a>-2147483648 else  2147483647 if a>0 else -2147483648)
else:
    print(0)