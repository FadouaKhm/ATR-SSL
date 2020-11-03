import re
import time
import os



def SearchString(Text,Pattern):
	match=re.search(Pattern,Text)
	if (match):
		return match.end()
	else:
		return -1

def ParseKeyValStringPair(s):
	#print(s)
	p=re.compile(r'[\S]+')
	match=p.search(s)
	if (match):
		#print(match)
		Key=match[0]
		#print(Key)
		ValStart=match.end()
	else:
		return True,{}
	p=re.compile(r'"[\S\s]+"')
	match=p.search(s[ValStart:])
	if (match):
		Val=match[0][1:-1]
		#print(Val)
		return False,{Key:Val}
	else:
		return True,{}

def ParseKeyFloatListPair(s):
	p=re.compile(r'[\S]+')
	match=p.search(s)
	if (match):
		#print(match)
		Key=match[0]
		#print(Key)
		ValStart=match.end()
	else:
		return True,{}
	#print(s[ValStart:])

	p=re.compile(r'[-]*[\d]+[.][\d]+')
	match=p.findall(s[ValStart:])
	return False,{Key:[float(x) for x in match]}

def ParseKeyIntListPair(s):
	p=re.compile(r'[\S]+')
	match=p.search(s)
	if (match):
		#print(match)
		Key=match[0]
		#print(Key)
		ValStart=match.end()
	else:
		return True,{}
	#print(s[ValStart:])

	p=re.compile(r'[-]*[\d]+')
	match=p.findall(s[ValStart:])
	return False,{Key:[int(x) for x in match]}


def ReduceDict(res):
	for x in res:
		if len(res[x])==1:
			res[x]=res[x][0]

def UpdateDict(FinalRes,res):
	for x in res:
		if x not in FinalRes:
			FinalRes[x]=[]
		FinalRes[x].append(res[x])


def ParseSenUpd(f):
	if SearchString(f.readline(),'{')==-1:
		return True,{}

	#print ('in ParseSenUpd')

	FinalRes={}

	while True:
		s=f.readline()
		#print(s)

		if SearchString(s,'Name')>=0 or \
		   SearchString(s,'Scenario')>=0 or \
		   SearchString(s,'Keyword')>=0 or \
		   SearchString(s,'Comment')>=0 :
			Er,res=ParseKeyValStringPair(s)
		elif SearchString(s,'Range')>=0 or \
			 SearchString(s,'Elevation')>=0 or \
			 SearchString(s,'Azimuth')>=0 or \
		     SearchString(s,'Aspect')>=0:
			Er,res=ParseKeyFloatListPair(s)
			if Er==True:
				return True, {}
			for x in res:
				if len(res[x])!=1:
					return True, {}	
				res[x]=res[x][0]
		elif SearchString(s,'Fov')>=0:
			Er,res=ParseKeyFloatListPair(s)
		elif SearchString(s,'Time')>=0:
			Er,res=ParseKeyIntListPair(s)
		elif  SearchString(s,'}')>=0 :
			#print ('out ParseSenUpd')
			#print(FinalRes)
			ReduceDict(FinalRes)
			return False,{'SenUpd':FinalRes}
		else:
			return True,{}
		
		if Er==False:
			#print(res)
			UpdateDict(FinalRes,res)
		else:
			return True,{}




def ParseSenSect(f):
	if SearchString(f.readline(),'{')==-1:
		return True,{}

	#print ('in ParseSenSect')

	SenUpdList=[]
	FinalRes={}

	while True:
		s=f.readline()


		if SearchString(s,'Name')>=0:
			Er,res=ParseKeyValStringPair(s)
		elif SearchString(s,'SenUpd')>=0:
			Er,res=ParseSenUpd(f)
		elif  SearchString(s,'}')>=0 :
			#print ('out ParseSenSect')
			#print(SenUpdList)
			ReduceDict(FinalRes)
			FinalRes['SenUpdList']=SenUpdList
			return False,{'SenSect':FinalRes}
		else:
			return True,{}
		
		if Er==False:
			#print(res)
			for x in res:
				if x=='SenUpd':
					SenUpdList.append(res)
				else:
					UpdateDict(FinalRes,res)
		else:
			return True,{}



def ParsePrjSect(f):
	if SearchString(f.readline(),'{')==-1:
		return True,{}

	#print('in ParsePrjSect')

	FinalRes={}

	while True:
		s=f.readline()

		if SearchString(s,'Name')>=0 or \
		   SearchString(s,'Scenario')>=0 or \
		   SearchString(s,'Keyword')>=0 or \
		   SearchString(s,'Comment')>=0 :
			Er,res=ParseKeyValStringPair(s)
		elif  SearchString(s,'}')>=0 :
			#print ('out ParsePrjSect')
			ReduceDict(FinalRes)
			return False,{'PrjSect':FinalRes}
		else:
			return True,{}
		if Er==False:
			#print(res)
			UpdateDict(FinalRes,res)
		else:
			return True,{}


def ParseTgt(f):
	if SearchString(f.readline(),'{')==-1:
		return True,{}

	#print("in ParseTgt")

	FinalRes={}

	while True:
		s=f.readline()

		if SearchString(s,'Comment')>=0 or \
		   SearchString(s,'Keyword')>=0 or \
		   SearchString(s,'TgtType')>=0 or \
		   SearchString(s,'PlyId')>=0 :
			Er,res=ParseKeyValStringPair(s)
		elif SearchString(s,'Range')>=0 or \
		     SearchString(s,'Aspect')>=0:
			Er,res=ParseKeyFloatListPair(s)
			if Er==True:
				return True, {}
			for x in res:
				if len(res[x])!=1:
					return True, {}	
				res[x]=res[x][0]
		elif SearchString(s,'PixLoc')>=0:
			Er,res=ParseKeyIntListPair(s)
			if Er==True:
				return True, {}
			if len(res['PixLoc'])!=2:
				return True, {}
			res['PixLoc']={'x':res['PixLoc'][0],'y':res['PixLoc'][1]}
		elif  SearchString(s,'}')>=0 :
			#print ('out ParseTgt')
			#print({'Tgt':FinalRes})
			ReduceDict(FinalRes)
			return False,{'Tgt':FinalRes}
		else:
			return True,{}
		if Er==False:
			#print(res)
			UpdateDict(FinalRes,res)
		else:
			return True,{}


def ParseTgtUpd(f):
	if SearchString(f.readline(),'{')==-1:
		return True,{}

	#print("in ParseTgtUpd")

	TgtList=[]
	FinalRes={}

	while True:
		s=f.readline()

		if SearchString(s,'Comment')>=0 or \
		   SearchString(s,'Keyword')>=0:
			Er,res=ParseKeyValStringPair(s)
		elif SearchString(s,'Time')>=0:
			Er,res=ParseKeyIntListPair(s)
		elif  SearchString(s,'Tgt')>=0 :
			Er,res=ParseTgt(f)
		elif  SearchString(s,'}')>=0 :
			#print ('out ParseTgtUpd')
			ReduceDict(FinalRes)
			FinalRes['TgtList']=TgtList
			return False,{'TgtUpd':FinalRes}
		else:
			return True,{}
		if Er==False:
			#print(res)
			for x in res:
				if x=='Tgt':
					TgtList.append(res)
				else:
					UpdateDict(FinalRes,res)
		else:
			return True,{}

def ParseTgtSect(f):
	if SearchString(f.readline(),'{')==-1:
		return True,{}

	#print("in ParseTgtSect")

	TgtUpdList=[]

	while True:
		s=f.readline()

		if SearchString(s,'TgtUpd')>=0:
			Er,res=ParseTgtUpd(f)
		elif  SearchString(s,'}')>=0 :
			#print ('out ParseTgtSect')
			return False,{'TgtSect':TgtUpdList}
		else:
			return True,{}
		
		if Er==False:
			#print(res)
			TgtUpdList.append(res)
		else:
			return True,{}



def ParseAGT(f):
	if SearchString(f.readline(),'{')==-1:
		return True,{}

	#print("in ParseAGT")

	FinalRes={}

	while True:
		s=f.readline()

		if SearchString(s,'PrjSect')>=0:
			Er,res=ParsePrjSect(f)
			#print(res)
		elif SearchString(s,'SenSect')>=0:
			Er,res=ParseSenSect(f)
		elif SearchString(s,'TgtSect')>=0:
			Er,res=ParseTgtSect(f)
		elif SearchString(s,'}')>=0:
			#print ('out ParseAGT')
			return False,{'Agt':FinalRes}
		else:
			return True,{}

		if Er==False:
			#print(res)
			for x in res:
				FinalRes[x]=res[x]
		if Er==True:
			return True,{}



def ParseAGTFile(FileName):
	f = open(FileName, "r")
	if SearchString(f.readline(),'Agt')==-1:
		return True,{}

	#print("in ParseAGTFile")

	Er,res=ParseAGT(f)
	if Er==True:
		return True,{}
	else:
		return Er,res

#print(os.listdir(r"cegr\agt"))
#FileList=os.listdir(r"cegr\agt")
#FileList=['cegr02001_0000.agt']
#for x in FileList:
#	start = time.time()
#	Er,res=ParseAGTFile("cegr\\agt\\"+x)
#	end = time.time()
#	print(x,Er,end-start)



