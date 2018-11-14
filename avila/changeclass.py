import collections

filename = 'avila-ts.txt'
out = open('avila-ts-01.txt', 'w')

with open(filename, 'r') as f:
	lines = f.readlines()
	
	for line in lines:
		#print line[-1], 1, line[-2], 2, line[-3]
		y = line[-3]
		out.write(line[:-3])
		out.write('0' if y == 'A' else '1')
		out.write('\n')
#
out.close()