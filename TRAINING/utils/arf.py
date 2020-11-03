#!/usr/bin/env python

import struct
import numpy as np
import os
import string

zlibunavail = 0
try:
    import SwapFile
except ImportError:
    zlibunavail = 1

# VM - changed exception handling from a string to a class
class ArfError(Exception):
    def __init__(self,value):
        self.value = value

# VM added type 6
typemap={0:(np.uint8,1,'eight_bit'), 1:(np.uint16,2,'ten_bit'),
         2:(np.uint16,2,'twelve_bit'), 5:(np.uint16,2,'short_word'),
         6:(np.uint32,4,'long_word'), 7:(np.float32,4,'float_single'),
         8:(np.float64,8,'float_double'), 10:(np.uint32,4,'multi_band'),
         13:(np.uint32,4,'hyper_band')}
rtypemap={'b':(0,1,'eight_bit'),'w':(5,2,'short_word'),'f':(7,4,'float_single'),'d':(8,8,'float_double') }
nametotype={}
MAGIC_NUM=0xBBBBBAAD
compressed_formats=('.gz', '.z')

for imagetype,(arr_typecode,elsize,name) in list(typemap.items()):
    nametotype[name]=imagetype

class ARF:
    def __init__(self):
        self.num_frames=0
        self.offset=32
        self.flags=0
        self.magic_num=MAGIC_NUM

    def load(self,frame,band=0):
        #print(self.offset, self.foots,"               ")
        if hasattr(self,"smb_boffs"):
            band      = band % self.smb_nbands
            offset    = self.smb_boffs[band] + int(frame)*self.frmsz
            num_bytes = self.smb_bsize[band]
            Num_type  = typemap[self.smb_btype[band]][0]
        else:
            offset    = self.offset + int(frame)*self.frmsz
            num_bytes = self.frmsz - self.foots
            Num_type  = typemap[self.image_type][0]
        #print(offset, num_bytes, self.image_type, Num_type)
        self.fh.seek(offset)
        buf=self.fh.read(num_bytes)
        arr=np.fromstring(buf,Num_type)
        if np.little_endian:
            #print("Swapping bytes")
            #arr=arr.byteswapped()
            arr=arr.byteswap()
        arr.shape=(self.num_rows,self.num_cols)
        return(arr)
        
    def get_frame_time(self, frame):
        if self.foots == 0: 
            return 'No footer information'

        num_bytes = self.offset + frame*self.frmsz - self.foots
        self.fh.seek(num_bytes)
        
        #only read first 40 bytes for now
        footer_buf = self.fh.read(40)
        (x,y,az,el,yy,dd,hh,mm,ss,ms )=struct.unpack('>10L',footer_buf)
        
        return (yy,dd,hh,mm,ss,ms )
    def save(self,arr):
        if len(arr.shape)!=2 or arr.typecode() not in 'bwfd':
            raise ArfError("array type not supported")

        if not hasattr(self,"num_rows"):
            self.num_rows=arr.shape[0]
        if not hasattr(self,"num_cols"):
            self.num_cols=arr.shape[1]
        if not hasattr(self,"image_type"):
            self.image_type=rtypemap[arr.typecode()][0]

        if arr.shape!=(self.num_rows,self.num_cols) or rtypemap[arr.typecode()][0]!=self.image_type:
            raise ArfError('array does not match ARF file attributes')

        if np.little_endian:
            arr=arr.byteswapped()

        # does this need to be changed to ravel?
        self.fh.write(np.ravel(arr).tostring())
        self.fh.flush()

        self.num_frames=self.num_frames+1
        if np.little_endian:
            arr=arr.byteswapped()

    def __del__(self):
        if self.mode=='w':
            if not hasattr(self,"num_rows"):
                print('ArfWarning: file opened for writing being closed without any images saved')
            else:
                self.fh.seek(0)
                hdr=struct.pack('>L7l',MAGIC_NUM,2,self.num_rows,self.num_cols,self.image_type,self.num_frames,32,0)
                self.fh.write(hdr)
        self.fh.close()

    def __str__(self):
        return(str(self.__dict__))

def iscompressed(filename):
    fname, ext = os.path.splitext(filename)
    if string.lower(ext) in (".z", ".gz", ".zip"):
        return 1
    return 0

def arf_open(filename,mode='r'):
    try:
       import builtins
    except ImportError:
       import __builtin__ as builtins

    if zlibunavail and filename[-2:] == 'gz':
        'Cannot open file - zlib not found'
        return

    if mode=='r':
        if not zlibunavail and iscompressed(filename):
            fh = SwapFile.GZSwapFile(filename, 'rb')
        else:
            fh=builtins.open(filename,'rb')

        #print "Before reading from header"
        hdr=fh.read(32)
        #print "After reading from header"

        (magic_num,ver,h,w,image_type,num_frames,offset,flags)=struct.unpack('>8l',hdr)
        #print "After calling struct.unpack"
        if magic_num!=-1145324883:
            raise ArfError('Bad ARF magic number')
        #print "after accessing array typemap"
        
        o=ARF()
        o.num_cols=w
        o.num_rows=h
        o.file_type=image_type
        image_type=typemap[image_type][2]
        o.image_type=nametotype[image_type]
        o.offset=offset
        o.num_frames=num_frames
        #print "o.image_type=",o.image_type
        if flags & 1:
            ts=fh.read(48)
            (sinf_imgsrc,sinf_st_x,sinf_st_y,sinf_n_avg,sinf_capt_rate,
                 sinf_y,sinf_dofy,sinf_hh,sinf_mm,sinf_ss,sinf_ms,strl)=struct.unpack('>12l',ts)
            if strl%4:
                strl = strl + (4-strl%4)
            sinf_cap_loc=fh.read(strl)
            ts=fh.read(4)
            (strl)=struct.unpack('>1l',ts)
            strl = strl[0]
            if strl%4:
                strl = strl + (4-strl%4)
            sinf_cap_loc=fh.read(strl)
            ts=fh.read(4)
            (strl)=struct.unpack('>1l',ts)
            strl = strl[0]
            if strl%4:
                strl = strl + (4-strl%4)
            sinf_digitizer=fh.read(strl)
            ts=fh.read(8)
            (sinf_fovx,sinf_fovy)=struct.unpack('>2f',ts)
            ts=fh.read(4)
            (strl)=struct.unpack('>1l',ts)
            sinf_sampperdwell = strl[0]
        if flags & 2:
            scolmap=fh.read(3072)
        if flags & 4:
            ts=fh.read(4)
            (strl)=struct.unpack('>1l',ts)
            strl = strl[0]
            if strl%4:
                strl = strl + (4-strl%4)
            scomment=fh.read(strl)
            print("\n",scomment,"\n")
        o.foots=0
        if flags & 8:
            ts=fh.read(4)
            (strl)=struct.unpack('>1l',ts)
            strl = strl[0]
            if strl%4:
                strl = strl + (4-strl%4)
            o.smb_name=fh.read(strl)
            ts=fh.read(4)
            (strl)=struct.unpack('>1l',ts)
            showband=1
            o.smb_nbands = strl[0]
            o.smb_bname=[]
            o.smb_btype=[]
            o.smb_bsize=[]
            o.smb_boffs=[]
            for i in range(16):
                ts=fh.read(4)
                (strl)=struct.unpack('>1l',ts)
                strl = strl[0]
                if strl%4:
                    strl = strl + (4-strl%4)
                if strl>0:
                    o.smb_bname.append(fh.read(strl))
                else:
                    o.smb_bname.append(i)
                ts=fh.read(4)
                (strl)=struct.unpack('>1l',ts)
                o.smb_btype.append(strl[0])
                o.smb_bsize.append(typemap[o.smb_btype[i]][1]*o.num_rows*o.num_cols)
                if i == 0:
                    o.smb_boffs.append(o.offset)
                elif i < o.smb_nbands:
                    o.smb_boffs.append(o.smb_boffs[i-1]+o.smb_bsize[i-1])
        if flags & 16:
            ts=fh.read(4)
            (strl)=struct.unpack('>1l',ts)
            sftrflgs = strl[0]
            if sftrflgs & 1: o.foots += 40
            if sftrflgs & 2: o.foots += 96
        if flags & 32:
            ts=fh.read(24)
            (sinf_fmscale,sinf_resrngbias,sinf_hifov,sinf_vifov,
                                  sinf_courseres,sinf_fineres) =struct.unpack('>6f',ts)
            sftrflgs = strl[0]
            print("\nfm_scale             ",sinf_fmscale)
            print("resolved_range_bias  ",sinf_resrngbias)
            print("hor_ifov             ",sinf_hifov)
            print("ver_ifov             ",sinf_vifov)
            print("coarse_res           ",sinf_courseres)
            print("fine_res             ",sinf_fineres,"\n")
        if flags & 64:
            ts=fh.read(4)
            (strl)=struct.unpack('>1l',ts)
            strl = strl[0]
            if strl%4:
                strl = strl + (4-strl%4)
            o.smb_name=fh.read(strl)
            ts=fh.read(4)
            (strl)=struct.unpack('>1l',ts)
            showband=1
            o.smb_nbands = strl[0]
            o.smb_bname=[]
            o.smb_btype=[]
            o.smb_bwide=[]
            o.smb_bhigh=[]
            o.smb_bsize=[]
            o.smb_boffs=[]
            for i in range(o.smb_nbands):
                ts=fh.read(4)
                (strl)=struct.unpack('>1l',ts)
                strl = strl[0]
                if strl%4:
                    strl = strl + (4-strl%4)
                if strl>0:
                    o.smb_bname.append(fh.read(strl))
                else:
                    o.smb_bname.append(i)
                ts=fh.read(12)
                (strl)=struct.unpack('>3l',ts)
                o.smb_btype.append(strl[0])
                o.smb_bwide.append(strl[1])
                o.smb_bhigh.append(strl[2])
                o.smb_bsize.append(typemap[o.smb_btype[i]][1]*o.smb_bwide[i]*o.smb_bhigh[i])
                print(o.smb_btype[i],o.smb_bwide[i],o.smb_bhigh[i],o.smb_bsize[i])
                if i == 0:
                    o.smb_boffs.append(o.offset)
                elif i < o.smb_nbands:
                    o.smb_boffs.append(o.smb_boffs[i-1]+o.smb_bsize[i-1])
        o.flags=flags
        if o.image_type == 10 or o.image_type == 13:
            o.frmsz = o.smb_boffs[o.smb_nbands-1] + o.smb_bsize[o.smb_nbands-1] - o.offset + o.foots
            o.image_type = o.smb_btype[0]
        else:
            o.frmsz = typemap[o.image_type][1]*o.num_rows*o.num_cols + o.foots
        if not hasattr(o,"smb_nbands"):
            o.smb_nbands = 1
        print("%s - ImgTyp/FrmSz/FooterSz = %s/%d/%d"%(filename,image_type,o.frmsz,o.foots))
    if mode=='w':
        if not zlibunavail and iscompressed(filename):
            fh = SwapFile.GZSwapFile(filename, 'wb')
        else:
            fh=builtins.open(filename,'wb')
        fh.seek(32)
        o=ARF()
    o.mode=mode
    o.filename=filename
    o.fh=fh
    return(o)

def create(filename,arr):
    try:
       import builtins
    except ImportError:
       import __builtin__ as builtins

    if not zlibunavail and iscompressed(filename):
        fh = SwapFile.GZSwapFile(filename, 'wb')
    else:
        fh=builtins.open(filename,'wb')

    image_type=rtypemap[arr.typecode()][0]
    if np.LittleEndian:
        arr=arr.byteswapped()
    num_rows,num_cols=arr.shape
    hdr=struct.pack('>L7l',MAGIC_NUM,2,num_rows,num_cols,image_type,1,32,0)
    fh.write(hdr)
    fh.write(arr.tostring())
    fh.close()
