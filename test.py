set1 = {1: {'abc': 3}}

for doc in set1:
  for term in set1[doc]:
    set1[doc][term] = 4

print(set1)