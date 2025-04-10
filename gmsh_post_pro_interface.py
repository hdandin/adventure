#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:05:34 2023

@author: nchevaug
"""
from __future__ import annotations
import gmsh
import numpy as np
import numpy.typing as npt

FNDArray = npt.NDArray[np.float64]

nbData2DataType       = {1:'scal', 2:'vec2d', 3:'vec3d', 4:'tens2d', 9:'tens3d' }
entityType2nbVertices = {'point':1,'edge':2, 'tri':3, 'quad':4, 'tet':4}
entityType2minDim     = {'point':1,'edge':1, 'tri':2, 'quad':2, 'tet':3}
entityType2gmshType   = {'point':'P','edge':'L', 'tri':'T'} #, 'quad':2, 'tet':3}
dataType2gmshType     = {'scal':'S', 'vec2d':'V', 'vec3d':'V', 'tens2d': 'T','tens3d':'T' }
gmshVectorDisplay     = {'Line':1, 'Arrow':2, 'Pyramid':3, '3d Arrow':4, 'Displacement':5, 'Comet':6}
gmshIntervalType      = {'Iso-Value':1, 'Continuous map':2, 'Filled iso-values':3, 'Numeric values':4}
gmshPointType         = {'Color dot':4, '3D sphere':1, 'Scaled dot':2, 'Scaled sphere':3}
gmshLineType          = {'Color segment':3, '3D cylinder':1, 'Tapered cylinder':2}


views = []


def gmshExportsViews(basefilename, extension, eviews = views ) :
    #print(views)
    if extension in ['mpg']:
        gmsh.write(basefilename+'.'+extension)
        return
    if extension == 'pos':
        if len(eviews)> 0:
            gmsh.view.write(views[0], basefilename+'.pos', append = False)
            for view in eviews[1:]:
                gmsh.view.write(view, basefilename+'.pos', append = True)
        gmsh.write(basefilename+'.opt')
        return
    for view in views : gmsh.view.setVisibilityPerWindow(view, 0)
    gmsh.option.setNumber('General.SmallAxes' , 0);
    digits = 1 if len(views) == 0 else int(np.log10(len(views))+1)
    for i, view in enumerate(views) :
        gmsh.view.setVisibilityPerWindow(view, 1)
        gmsh.write(basefilename+str(i).zfill(digits)+'.'+extension)
        gmsh.view.setVisibilityPerWindow(view, 0)
    for view in views : gmsh.view.setVisibilityPerWindow(view, 1)
    gmsh.option.setNumber('General.SmallAxes' , 1);

def gmshListPlotFormatEntities(xyz_:FNDArray, entityType:str, nEntities = None) :
    inShapexyz_ = xyz_.shape
    fname = gmshListPlotFormatEntities.__name__
    def raiseGmshFormat() :
        raise ValueError("Can't interprete the xyz in "+fname)
    nbv   = entityType2nbVertices.get(entityType)
    if nbv is None :
        raise ValueError("entityType "+entityType+"is unknown in "+fname)
    if nEntities is None :
        if len(xyz_.shape) == 1: raiseGmshFormat()
        nEntities = xyz_.shape[0]
    if len(xyz_.shape) == 1 :
        if (xyz_.shape[0]%nEntities) : raiseGmshFormat()
        xyz_.shape = (nEntities, -1)
    if len(xyz_.shape) == 2 :
        minDim = entityType2minDim[entityType]
        if xyz_.shape[1] == nbv :
            if minDim > 1 : raiseGmshFormat()
            xyz_.shape = (nEntities, nbv, 1) 
        elif xyz_.shape[1] == 2*nbv :
            if minDim > 2 : raiseGmshFormat()
            xyz_.shape = (nEntities, nbv, 2) 
        elif xyz_.shape[1] == 3*nbv:
            xyz_.shape = (nEntities, nbv, 3)
        else : raiseGmshFormat()
    if xyz_.shape[:2] != (nEntities, nbv) : raiseGmshFormat()
    if len(xyz_.shape) != 3 :
        raise ValueError("xyz should be of ndim atleast 3")
    ncoord = xyz_.shape[2]
    if ncoord < 3 :
        #filling up yz coordinates with zeros if needed
        z = np.zeros(xyz_.shape[:-1]+(3-ncoord,))
        xyz = np.concatenate((xyz_,z),2)
    elif ncoord == 3 :
        xyz = xyz_.copy()
    else : raiseGmshFormat()
    xyz =  np.moveaxis(xyz, 1,2)
    # return shapes to there input values
    xyz_.shape = inShapexyz_
    # return result
    return nEntities, xyz

def gmshListPlotFormatDataElem2ElemNode(constantData:FNDArray, nEntities, entityType) -> FNDArray:
    nbv = entityType2nbVertices[entityType]
    return np.repeat(constantData.reshape((nEntities, -1)), nbv, axis = 0).reshape((nEntities, nbv, -1))

def gmshListPlotFormatData(data_:FNDArray, nEntities:int, entityType:str, P0 =False) -> tuple[str, int, FNDArray] :
    fname = gmshListPlotFormatData.__name__
    errorstr = "can't interprete the data_ in "+fname
    inShapedata_= data_.shape
    indata_ = data_
    if P0:
        data_ = gmshListPlotFormatDataElem2ElemNode(data_, nEntities, entityType)
    data_type = 'unknown'
    nbv = entityType2nbVertices[entityType]
    if data_.ndim > 4: raise ValueError(errorstr)
    if data_.ndim == 1 :
        if data_.size%nEntities == 0 : data_.shape=(nEntities, -1)
        else : raise ValueError(errorstr)
    if data_.shape[0] != nEntities : raise ValueError(errorstr)
    if data_.ndim == 2 :
        if data_.shape[1]%nbv != 0: raise ValueError(errorstr)
        data_.shape = (nEntities, nbv, -1)
    if data_.ndim == 3 :
        nbdata = data_.shape[2]
        data_type = nbData2DataType.get(nbdata, None)
        if data_type is None:
            raise ValueError(errorstr)
        if data_type == 'tens2d' :
            data_.shape = (nEntities, nbv, 2, 2)
        if data_type == 'tens3d' :
            data_.shape = (nEntities, nbv, 3, 3)
    elif data_.ndim == 4:
        if data_.shape[2:] == (2,2) :
            data_type = 'tens2d'
        elif data_.shape[2:] == (3,3) :
            data_type = 'tens3d'
        else : raise ValueError(errorstr)
    # filling up data with zeros where needed
    if data_type == 'vec2d':
        data = np.zeros((nEntities, nbv, 3))
        data[:,:,0:-1] = data_
    elif data_type == 'tens2d':
        data = np.zeros((nEntities, nbv, 3, 3))
        data[:,:,0:-1, 0:-1] = data_
    else :
        data = data_.copy()
    # return shapes to there input values
    indata_.shape  = inShapedata_
    # return result
    return  data_type, nEntities, data

def gmshListPlotFormatEntityData(entityVerticesCoords, data, entityType, nEntities = None, P0 = False):
    nEntities, entityVerticesCoords = gmshListPlotFormatEntities(entityVerticesCoords, entityType, nEntities)
    dataType, __, data = gmshListPlotFormatData(data, nEntities, entityType, P0)
    data = np.column_stack((entityVerticesCoords.reshape((nEntities,-1)), data.reshape((nEntities,-1)))).flatten()
    gmshDataTypeEntityType = dataType2gmshType[dataType] + entityType2gmshType[entityType]
    return gmshDataTypeEntityType, nEntities, data

def setViewOption(gv, IntervalsType = "Filled iso-values", NbIso = 20,  VectorType = "Arrow", DisplacementFactor = 1., Range = None) :
    gmsh.view.option.setNumber(gv, "IntervalsType", gmshIntervalType.get(IntervalsType, 3))
    gmsh.view.option.setNumber(gv, "NbIso", NbIso)
    gmsh.view.option.setNumber(gv, "VectorType", gmshVectorDisplay.get(VectorType, 2))
    gmsh.view.option.setNumber(gv, "DisplacementFactor",    DisplacementFactor)
    if Range is not None :
        gmsh.view.option.setNumber(gv, "RangeType",  2)
        gmsh.view.option.setNumber(gv, "CustomMin", Range[0])
        gmsh.view.option.setNumber(gv, "CustomMax", Range[1])
    return gv

def listPlotFieldPoint(xyz, data, viewname = 'T', nPoints = None, PointType = "3D sphere", PointSize = 5, IntervalsType = "Filled iso-values", NbIso = 20,  VectorType = "Arrow", DisplacementFactor = 0., Range = None, gv = None):
    if gv is None : 
        gv = gmsh.view.add(viewname)
        views.append(gv)
    gmsh.view.addListData(gv, *gmshListPlotFormatEntityData(xyz, data, 'point', nPoints) )
    gmsh.view.option.setNumber(gv,"PointType",     gmshPointType.get(PointType, 2))
    gmsh.view.option.setNumber(gv,"PointSize",     PointSize)
    setViewOption(gv, IntervalsType = IntervalsType, NbIso = NbIso, VectorType = VectorType, DisplacementFactor = DisplacementFactor, Range = Range)
    return gv

def listPlotFieldLine(lineVerticesCoord, data, viewname = 'T', P0 = False, nLines = None, LineType = '3D cylinder', LineWidth = 2, IntervalsType = "Filled iso-values", NbIso = 20,  VectorType = "Arrow", DisplacementFactor = 0., Range = None, gv = None):
    if gv is None :
        gv = gmsh.view.add(viewname)
        views.append(gv)
    gmsh.view.addListData(gv, *gmshListPlotFormatEntityData(lineVerticesCoord, data, 'edge', nLines, P0))
    gmsh.view.option.setNumber(gv, "LineType",  gmshLineType.get(LineType, 2))
    gmsh.view.option.setNumber(gv, "LineWidth", LineWidth)
    gmsh.view.option.setNumber(gv, "GlyphLocation", 1 if P0 else 2) 
    setViewOption(gv, IntervalsType = IntervalsType, NbIso = NbIso, VectorType = VectorType, DisplacementFactor = DisplacementFactor, Range = Range)
    return gv 

def listPlotFieldTri(trisVerticesCoord, trisData, viewname ='T', P0 = False, ntri = None,  IntervalsType = "Filled iso-values", NbIso = 20, VectorType = "Arrow", DisplacementFactor = 0., Range = None, gv =None):
    if gv is None  :
        gv = gmsh.view.add(viewname)
        views.append(gv)
    gmshDataTypeEntityType, ntri, data = gmshListPlotFormatEntityData(trisVerticesCoord, trisData, 'tri', ntri, P0)
    gmsh.view.addListData(gv, gmshDataTypeEntityType, ntri, data)
    gmsh.view.option.setNumber(gv,"ShowElement", 0 if ntri > 10000 else 1)
    gmsh.view.option.setNumber(gv, "GlyphLocation", 1 if P0 else 2) 
    setViewOption(gv, IntervalsType = IntervalsType, NbIso = NbIso, VectorType = VectorType, DisplacementFactor = DisplacementFactor, Range = Range)
    return gv
  
if __name__ == '__main__':
    testTri    = True
    testLine   = True
    testPoint  = True
    popUpGmsh  = True
    gmsh.initialize()
    xy = np.array([[0.,0.], [1.,0.], [1.,1.], [0.,1.]])
    xyz = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,1.], [0.,1.,0.]])
    if testPoint :
        data = np.array([0.,1.,2.,3.])
        listPlotFieldPoint(xy, data, viewname = 'Point xy Scalar', nPoints = None, PointType = '3D sphere', PointSize = 5, IntervalsType = "Filled iso-values",  VectorType = "Arrow", gv = None)
        listPlotFieldPoint(xyz, data, viewname = 'Point xyz, Scalar', nPoints = None, PointType = 'Color dot', PointSize = 5, IntervalsType = "Filled iso-values", VectorType = "Arrow", gv = None)
        data = np.array([1.,0.,0.,1., -1.,0., 0., -1.])
        listPlotFieldPoint(xy, data, viewname = 'Point xy vector2d', nPoints = None, PointType = 'Scaled dot', PointSize = 5, IntervalsType = "Filled iso-values", VectorType = "Arrow", gv = None)
        listPlotFieldPoint(xyz, data, viewname = 'Point xyz, vector2d', nPoints = None, PointSize = 5, IntervalsType = "Filled iso-values", VectorType = "Arrow", gv = None)
        data = np.array([1.,0.,0., 0.,1.,1., -1.,0., -1., 0., -1., 0.])
        listPlotFieldPoint(xy, data, viewname = 'Point xy vector3d', nPoints = None, PointSize = 5, IntervalsType = "Filled iso-values", VectorType = "Arrow", gv = None)
        listPlotFieldPoint(xyz, data, viewname = 'Point xyz, vector3d', nPoints = None, PointSize = 5, IntervalsType = "Filled iso-values",VectorType = "Comet", gv = None)
    if testLine :
       lines       = np.array([[0,1],[1,2],[2,3], [3,0]])
       xyLine      = xy[lines]
       xyzLine     = xy[lines]
       scalDataP1  = np.array([[0,1],[1,2], [2,3], [3,4]])
       scalDataP0  = np.array([0, 1, 2, 3])
       vectDataP0  = np.array([[0,-1],[1,0], [0,1], [-1,0]])
       listPlotFieldLine(xyzLine, scalDataP1, viewname = 'Line xyz scalar P1')
       listPlotFieldLine(xyzLine, scalDataP0, P0 =True, viewname = 'Line xyz scalar P0', LineType="Tapered cylinder")
       listPlotFieldLine(xyzLine, vectDataP0, P0 =True, viewname = 'Line xyz vector P0', LineType="Tapered cylinder")
    if testTri :
        triCoord = np.array([[[0.,0.],[1.,0.], [0., 1.]],[[1.,0.], [1.,1.], [0.,1.]]])
        #test tri scalar field
        triScalData = np.array([[0.,1.,2], [1.,3.,2]])
        listPlotFieldTri(triCoord, triScalData, viewname ='scalar Tri test', NbIso = 10, Range = None, gv =None)
        #test tri vector 2d field
        triVec2dData = np.array([[[0.,0.], [1.,0.], [0.,1]], [[1.,0.], [1.,1.], [0.,1.]]])
        listPlotFieldTri(triCoord, triVec2dData, viewname ='vecteur2D Tri test', NbIso = 10, Range = None, gv =None)
        #test tri vector 3d field
        triVec3dData = np.array([[[0.,0.,1.], [1.,0.,1.], [0.,1., 1.]], [[1.,0.,1.], [1.,1.,1.], [0.,1.,1.]]])
        listPlotFieldTri(triCoord, triVec3dData, viewname ='vecteur3D Tri test', NbIso = 10, Range = None, gv =None)
        #test tri tensor 2d  Flatten field
        triTens2dData = np.array([[[1.,0., 0., 0.], [0.,1.,0.,0.], [0.,0.,1.,0.]], [[0.,1.,0.,0.], [0.,0.,0.,1.], [0.,0.,1.,0.]]])
        listPlotFieldTri(triCoord, triTens2dData, viewname ='tensor2D flat Tri test', NbIso = 10, Range = None, gv =None)
        #test tri tensor 2d  Flatten field
        triTens2dData = np.array([[[[1.,0.], [0., 0.]], [[0.,1.],[0.,0.]], [[0.,0.],[1.,0.]]], [[[0.,1.],[0.,0.]], [[0.,0.],[0.,1.]], [[0.,0.],[1.,0.]]]])
        listPlotFieldTri(triCoord, triTens2dData, viewname ='tensor2D Tri test', NbIso = 10, Range = None, gv =None)  
    gmsh.option.setNumber('PostProcessing.AnimationCycle', 1)
    if popUpGmsh: gmsh.fltk.run() #start gmsh gui
    gmsh.finalize() #close gmsh
    