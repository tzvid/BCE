eppchs = [50, 60, 70, 80, 90]
num_crops_list = [1, 5, 10, 15, 20]

f = open('test.sh', 'w')
for e in eppchs:
    for num_crops in num_crops_list:
        f.write(f'python test_student.py --load-epoch {e} --num-crops {num_crops}\n')
        f.write(f'python test_student.py --load-epoch {e} --num-crops {num_crops} --BCE\n')

f.close()