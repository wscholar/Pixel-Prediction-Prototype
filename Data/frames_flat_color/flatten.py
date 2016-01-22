import os
import numpy
import scipy
import sys
from scipy import misc

path = "/Users/waynescholar/Downloads/mp4_samples"
dirs = os.listdir(path)

for file in dirs:
	if file.endswith('.png'):
		print file
		x=scipy.misc.imread(file)
		print x
		print x.shape
		numpy.savetxt(file.replace(".png",".txt"),x.ravel())
		

mysqlbinlog  mysql-bin.000061 |grep -i  'drop\|alter' | tr ‘[A-Z]’ ‘[a-z]’|sed -e '/*/d' | sort | uniq -c | sort -nr

agpb_phone,agpb_fname,agpb_lname,var1,salesperson,var2,var3
skipfirstrow:var3,id,var2,agpb_lname,agpb_fname,phone,var3,var4,var5,var6

skipfirstrow:agpb_fname,agpb_lname,agpb_phone,var1,id,var2

skipfirstrow:splitname:countdups(altphone):nosplit(balfouroutput):unique(altphone):req(description#Work Order):tic(description#WO#):skipthiscolumn,req(description#Property):tic(description#Property):skipthiscolumn,req(description#Unit):tic(description#Unit):skipthiscolumn,req(description#Priority):tic(description#Priority):skipthiscolumn,req(category):tic(category):skipthiscolumn,req(description#Brief Desc):tic(description#Brief Desc):skipthiscolumn,req(description#Full Desc):tic(description#Full Desc):skipthiscolumn,req(description#Problem Notes):tic(description#Problem Notes):skipthiscolumn,req(description#Tech Notes/Desc):tic(description#Tech Notes/Desc):skipthiscolumn,req(description#Call Date):tic(description#Call Date):skipthiscolumn,req(description#Schedule Date):tic(description#Schedule Date):skipthiscolumn,req(description#Complete Date):tic(description#Complete Date):skipthiscolumn,req(description#Caller Name):tic(description#Caller Name):skipthiscolumn,req(description#Caller Phone):tic(description#Caller Phone):agpb(agpb_phone),agpb(agpb_fname),agpb(var1),agpb(altphone),agpb(cell)

delete:needsphone:var2,id,var3,var3,agpb_lname,agpb_address,var4,var5,agpb_phone,var6,var6,var6,company,var7,var7,var1,var8