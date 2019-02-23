import csv
import re


# Create Rowdict to better index


rowdict = {}
with open("webpages.csv",encoding = "utf-8") as f:
	loading = csv.reader(f,delimiter = ',')
	for i,row in enumerate(loading):
		rowdict[i] = row

# A regular expression example 

# expre =  r'\b(\w+\.?\w*)\s*(?:\@+|\s+\W?at\W?\s+)\s*(\w+)\s*(?:\W+dot\W+|\.)\s*(\w+(?:\W*dot\W*|\.)?\w+)'
# for i in range(90,95):
# 	for string in rowdict[i]:
# 		prog = re.compile(expre)
# 		email = prog.search(string)
# 		if email:
# 			print(email.group(4))
# 		else:
# 			print('None')

# Create Function



def Find_email(string):
	expre = r'(\w+\.?\w*|[^>\s\@])\s*(?:\@+|\s+\W?at\W?\s+)\s*(\w+)\s*(?:\W+dot\W+|\.)\s*(\w+(?:\W*dot\W*|\.)?\w+)'
	prog = re.compile(expre)
	email = prog.search(string)
	if email:

		return '{}@{}.{}'.format(email.group(1),email.group(2),email.group(3))
	else:
		return  'None'
	


# Test function
# with open('sample_out.csv',encoding = 'utf-8', mode = 'w') as f:
# 	fieldnames = ['html', 'email']
# 	Headerwriter = csv.DictWriter(f, fieldnames=fieldnames)
# 	Headerwriter.writeheader()
# 	for i in range(5):
# 		string = rowdict[i][0]
# 		result = Find_email(string)
# 		print(result)
# 		writer = csv.writer(f)
# 		writer.writerow([string,result])

# Real try
with open("webpages.csv",encoding = "utf-8") as f:
	loading = csv.reader(f,delimiter = ',')
	with open('email_outputs.csv',encoding = 'utf-8',mode = 'w') as f1:
		fieldnames = ['html', 'email']
		Headerwriter = csv.DictWriter(f1, fieldnames=fieldnames)
		Headerwriter.writeheader()
		for i,row in enumerate(loading):
			if i > 0:
				string = row[0]
				result = Find_email(string)
				writer = csv.writer(f1)
				writer.writerow([string,result])

				

		
















