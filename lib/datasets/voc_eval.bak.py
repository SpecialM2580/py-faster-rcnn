# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import cPickle
import numpy as np
import pprint

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    #tree = ET.parse(filename)
    objs = [open(filename,'r')]
    objects = []
    for ix,obj in enumerate(objs):
        for t in obj.read().split('\n'):
            if t == '':
                continue
            #print t
            bbox = [t for t in t.split(';')]
            obj_struct=None
            obj_struct = {}
            #obj_struct['name'] = obj.find('name').text
            #obj_struct['pose'] = obj.find('pose').text
            #obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int('0')
            #bbox = obj.find('bndbox')
            obj_struct['bbox'] = [float(bbox[1]) ,
                              float(bbox[2]) ,
                              float(bbox[3]) ,
                              float(bbox[4]) ]
            clsnum= int(bbox[5])
            '''
            if clsnum in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]:
                obj_struct['name']='prohibitory'
                objects.append(obj_struct)
            elif clsnum in [33, 34, 35, 36, 37, 38, 39, 40]:
                obj_struct['name']='mandatory'
                objects.append(obj_struct)
            elif clsnum in [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
                obj_struct['name']='danger'
                objects.append(obj_struct)
            '''
            obj_struct['name']='sign'
            objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    use_07_metric=False
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))   #? 
        print 'mrec='
        pprint.pprint(mrec) 
        print 'mpre'
        pprint.pprint(mpre)
        for i in range(len(mrec)):
            plt.plot(mrec[i],mpre[i],'b*')
        plt.show()
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    #print('imagenames',len(imagenames))
    printnum=0;
    for imagename in imagenames:
        #[pprint.pprint(obj) for obj in recs[imagename]]
        for obj in recs[imagename]:
            print obj
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        #pprint.pprint(classname)
        bbox = np.array([x['bbox'] for x in R])
        #print [x['difficult'] for x in R]
        #difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        difficult=np.array(0 for x in R).astype(np.bool)
        det = [False] * len(R)
        #print('len=',len(R))
        printnum=printnum+len(R)
        #print('det=',len(det))
        npos = npos + len(R) #+ sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    #pprint.pprint(sorted_ind)
    sorted_scores = np.sort(-confidence)
    pprint.pprint(sorted_scores)
    
    BB = BB[sorted_ind, :]
    #image_ids = [image_ids[sorted_ind[x]] for x in range(len(sorted_ind)) if sorted_scores[x]<-0.650]
    image_ids=[image_ids[x] for x in sorted_ind]
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    print("nd=",nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        #print(d)
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            #if not R['difficult'][jmax]:
            if not R['det'][jmax]:
                #print("true=",image_ids[d])
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                #print("false d=",image_ids[d])
                fp[d] = 1.
        else:
            #print("BB=",BBGT[:, 0],BBGT[:, 1],BBGT[:, 2],BBGT[:, 3])
            #print("bb=",bb[0],bb[1],bb[2],bb[3])
            #print("false d=",image_ids[d]," ovmax=",ovmax)
            fp[d] = 1.
    # compute precision recall
    print("tp=",tp)
    print("fp=",fp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    print('npos=',float(npos))
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    #print('tp',tp)
    print('rec=',rec)
    print('prec=',prec)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
