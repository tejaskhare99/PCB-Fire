a = [[1,2,3,4],[22,43,54,34]]  #orignal coordinates (design)

b = [[3,5,6,3],[53,38,52,31]]  #checking for missing components

flag = []

for A in range(len(a)):
  flag.append('not_found')


for x in range(len(a)):
  for y in range(len(b)):
    a_ar = a[x]
    b_ar = b[y]
    a_xc = a_ar[0]
    a_yc = a_ar[1]
    a_hc = a_ar[2]
    a_wc = a_ar[3]

    b_xc = b_ar[0]
    b_yc = b_ar[1]
    b_hc = b_ar[2]
    b_wc = b_ar[3]

    if a_xc-5<= b_xc <= a_xc+5:
      if a_yc-5<= b_yc <= a_yc+5:
        if a_hc-5<= b_hc <= a_hc+5:
          if a_wc-5<= b_wc <= a_wc+5:
            flag[x]='found'

    else:
      pass


for l in range(len(flag)):
  if(flag[l]=='not_found'):
    print('element number {} is not found'.format(l+1))




