#此文件记录所需要用的所有函数
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import cartopy.crs as ccrs
from matplotlib.offsetbox import AnchoredText#关于修改text位置的
from skimage.feature import peak_local_max#寻找矩阵局部极值点
# 用于提取点集凹边界
import alphashape
import cartopy.feature as cfeature
#计算任意地球上多边形面积
from area import area
from scipy.interpolate import UnivariateSpline#三次样条平滑
from scipy.interpolate import griddata
from sklearn.neighbors import BallTree
import json
import matplotlib.path as mpath
#一、涡旋识别
##################################################################################################################################################################
#多图绘制首先需要创建坐标区，这里只需要传入两个参数，nrow:是你要绘制的行，fig指定坐标区在哪张图上绘制,num一行绘制几个
def creat_axes(nrow,fig,num,proj):
    x_interval,y_interval = 0.4,0.37
    y = 1.5-nrow*y_interval
    axs = [fig.add_axes([0.1+x_interval*i, y, 0.4, 0.3],projection = proj) for i in range(num)]
    caxs = [fig.add_axes([0.1+x_interval*i+0.02, y-0.035, 0.36, 0.01]) for i in range(num)]
    return axs,caxs
#判断两个list是否相同---------------------------------------------------------------
def islist_qual(list1,list2):
    list1 = np.sort(list1).tolist()
    list2 = np.sort(list2).tolist()
    return json.dumps(list1) == json.dumps(list2)
#判断list1是否都在list2中-------------------------------------------------------------
def islist_contains(list1,list2):
    list1 = np.array(list1)
    list2 = np.array(list2)
    return sum(np.isin(list1,list2))==len(list1)
#粗略计算格点间d的实际距离，支持一对多----------------------------------------------------------------------------------------------------------------------------
def geodistance(lon1,lat1,lon2,lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2]) # 经纬度转换成弧度
    dlon=lon2-lon1
    dlat=lat2-lat1
    a=np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distance=2*np.arcsin(np.sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=np.round(distance,3)
    return distance/1000

#将实际地理移动距离转换为经纬度，支持向量操作，这里考虑了经向移动中纬度间实际距离的变化----------------------------------------------------------------------------
def geo2lonlat_uv(lon, lat, u, v, t=3600):
    # 地球平均半径, 单位: 米
    R = 6378137
    lon,lat = np.radians(lon),np.radians(lat)
    pred_lat = np.rad2deg(lat+(v / R) * t)
    pred_lon = np.rad2deg(lon + (u / v) * np.arctanh(np.sin(lat+(v / R) * t)) - (u / v) * np.arctanh(np.sin(lat)))
    return pred_lon,pred_lat

#找到以某个点为中心，r距离内的格点---------------------------------------------------------------------------------------------------------------------------------
def find_r_points(r,central_lon,central_lat,lons,lats):
    lon_grid,lat_grid = np.meshgrid(lons.values,lats.values,indexing = 'xy')
    point_lons , point_lats = np.ones(len(lons)*len(lats))*central_lon,np.ones(len(lons)*len(lats))*central_lat
    distance_matrix = geodistance(lon1=point_lons,lat1=point_lats,lon2=lon_grid.reshape(-1),lat2= lat_grid.reshape(-1)).reshape((len(lats),len(lons)))
    #print(*zip(lon_grid[np.where(distance_matrix<r,True,False)],lat_grid[np.where(distance_matrix<r,True,False)]))
    return np.where(distance_matrix<r,True,False)

#识别边缘涡旋，不处理容易出bug----------------------------------------------------------------------------------------------------------------------------------------
def judge_frontier(original_vort_line,lons,lats):
    s,n = original_vort_line.vertices[:,1].min(),original_vort_line.vertices[:,1].max()
    w,e = original_vort_line.vertices[:,0].min(),original_vort_line.vertices[:,0].max()
    return sum([(s-lats.values.min())**2<0.1,(n-lats.values.max())**2<0.1,(w-lons.values.min())**2<0.1,(e-lons.values.max())**2<0.1])

#利用之前创建的经纬度查找树，寻找最邻近点并平滑------------------------------------------------------------------------------------------------------------------------
def smooth(data,lat_lon_points,lat_lon_tree,r):#r:km
    in_r_point = lat_lon_tree.query_radius(np.radians(lat_lon_points), r=r/6371)
    smooth_data = np.array([np.nanmean((data.values).reshape(-1)[in_r_point[i]]) for i in range(len(in_r_point)) ]).reshape(data.values.shape)    
    return smooth_data

#1、找到涡旋极值
def find_vort_min(smoothed_data,init_vort_min_line,temp_peaks,lons,lats,parameter_sets):
    smooth_r,vort_max0,useless_ax,vort_min0,scal_factor= parameter_sets
    #找到关键的vort_min线------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    lon_grid,lat_grid = np.meshgrid(lons.values,lats.values,indexing = 'xy')
    inline_point_bool = init_vort_min_line.contains_points(np.vstack([lon_grid.reshape(-1),lat_grid.reshape(-1)]).T).reshape((len(lats),len(lons)))
    data = smoothed_data.copy() #将线外的点置于0
    data[~inline_point_bool] = 0
    if len(temp_peaks)>1:
        secondary_max = np.sort(temp_peaks.peak_value.values)[-2]+0.2
    #反复缩小等值线delta直到所有峰值都可以被识别出
    peak_indexs = np.array([str(i) for i in temp_peaks.index])
    delta,count = 0.01,0
    while count<3:
        vort_min_levels = np.arange(vort_min0,secondary_max,delta)
        #以delta为间隔以第二峰值为最高，绘制等值线，找到vort_min线
        chosed_vort_min_lines = useless_ax.contour(lons,lats,data,levels=vort_min_levels,extend='both',colors = 'k',linestyles='--',linewidths=1) #画边界
        record = []
        for line_index in range(len(chosed_vort_min_lines.collections)):
            lines = chosed_vort_min_lines.collections[line_index].get_paths()
            for temp_line in lines:
                #找到每个等值线内包含的峰值，将包含的峰值编号转为字符串，并与对应的vort_min值和线记录
                inline_index_temp = np.sort(temp_peaks.loc[temp_line.contains_points(temp_peaks.loc[:,['lon','lat']].values)].index)
                inline_index_temp = ','.join([str(i) for i in inline_index_temp])
                record.append([inline_index_temp,vort_min_levels[line_index],temp_line])
        line_record = pd.DataFrame(record,columns=['inline_index','vort_min','vort_min_line'])
        
        identify_peaks = sum(np.isin(peak_indexs,np.unique(line_record.inline_index)))
        if identify_peaks==len(peak_indexs):
            break
        count+=1
        delta = delta/10
    if count ==3:
        #当存在峰值无法找到vort_min线时，忽略该峰值-------
        print('存在峰值无法识别')
        drop_indexs = peak_indexs[~np.isin(peak_indexs,np.unique(line_record.inline_index))].astype('int')
        for i in line_record.index:
            inline_index = line_record.loc[i,'inline_index']
            if inline_index!='':
                inline_index_array = np.array(inline_index.split(',')).astype('int')
                dropped_inline_index_list = inline_index_array[~np.isin(inline_index_array,drop_indexs)]
                line_record.loc[i,'inline_index'] = ','.join([str(j) for j in dropped_inline_index_list])
    #处理vort_min线，分出单多峰值情况------------------------------------------------------------------------------------------------
    vort_min_index = []
    for inline_index in line_record.inline_index.unique():
        if inline_index!='':
        #找到包含相同峰值的最大的等值线-----------------------------------------------
            vort_min_index.append(line_record.loc[line_record.inline_index==inline_index,'vort_min'].idxmin())
    vort_min_record = line_record.loc[vort_min_index].copy()
    vort_min_record['inline_index'] = [np.array(i.split(',')).astype('int') for i in vort_min_record.inline_index]
    vort_min_record['peak_num'] = [len(i) for i in vort_min_record.inline_index]
    vort_min_record.vort_min = np.floor(vort_min_record.vort_min*100)/100
    #将所有识别出来的单峰值（叶子节点）提取出来,合并得到初始边界信息----------------------------------------------------------------------------
    leaf = vort_min_record.loc[vort_min_record.peak_num==1].copy().drop(['peak_num'],axis=1)
    leaf_indexs = [i[0] for i in leaf.inline_index]
    leaf['inline_index']  = leaf_indexs
    leaf = pd.concat([temp_peaks.loc[leaf_indexs],leaf.set_index('inline_index')],axis=1)
    leaf['original_vort_line'] = init_vort_min_line
    leaf['is_isolated'] = False
    return vort_min_record,leaf
#3、对每一个original_line进行判断，计算极值中的真正涡旋点，以及边界 '''核心程序'''---------------------------------------------------------------------------------------------------------------------
def find_main_leaf(chosed_original_line,temp_peaks,vort_min_record,leaf,parameter_sets):
    #消去因峰值无法识别时已经被提出的峰值编号
    temp_peaks = temp_peaks.loc[np.isin(temp_peaks.index,leaf.index)]
    smooth_r,vort_max0,useless_ax,vort_min0,scal_factor = parameter_sets
    chosed_original_equal_r = chosed_original_line.equal_r
    #当半径小于120KM默认只有一个峰值
    if chosed_original_equal_r<120:
        max_index = temp_peaks['peak_value'].idxmax()
        leaf.loc[max_index,'vort_min'] = vort_min0
        leaf.loc[max_index,'vort_min_line'] = chosed_original_line.line_path
        leaf.loc[max_index,'is_isolated'] = True
        leaf['isolated_num'] = 1
        return leaf
    #主涡旋编号
    max_peak_index = temp_peaks['peak_value'].idxmax()
    #主涡旋经纬度
    max_peak_lon,max_peak_lat = temp_peaks.loc[max_peak_index,['lon','lat']]
    #判断每个vort_min_line的最大涡度峰值编号
    line_max_indexs = []
    for inline_index in vort_min_record.inline_index.values:
        line_max_indexs.append(temp_peaks.loc[inline_index,'peak_value'].idxmax())
    vort_min_record['max_peak_index'] = line_max_indexs
    #判断每个vort_min_line的最大涡度峰值以及离主涡旋的距离
    vort_min_record['max_peak_value']  = temp_peaks.loc[line_max_indexs,'peak_value'].values
    #vort_min_record['max_peak_distance'] = geodistance(max_peak_lon,max_peak_lat,temp_peaks.loc[line_max_indexs,'lon'],temp_peaks.loc[line_max_indexs,'lat']).values
    #距离判据以及强度判据,判断每个vort_min线包围的是否是独立的涡旋区域
    intensity_flag = (vort_min_record['max_peak_value']-vort_min_record['vort_min'])>vort_min_record['max_peak_value']*scal_factor
    #distance_flag = vort_min_record['max_peak_distance']>(chosed_original_line_l*distance_factor)
    #vort_min_record['is_isolated'] = np.logical_and(intensity_flag,distance_flag)
    vort_min_record['is_isolated'] = intensity_flag
    #包含主涡旋的涡旋区域须为独立涡旋
    vort_min_record.loc[vort_min_record.max_peak_index==max_peak_index,'is_isolated']=True
    #按照vort_min从高到低排列
    vort_min_record = vort_min_record.sort_values(by='vort_min',ascending=False)
    #选择独立涡旋
    isolated_vort_min_record = vort_min_record.loc[vort_min_record.is_isolated]
    #更新leaf，从高到低对vort_min进行判断，当该vort_min_line中的独立峰值数小于等于1，则其中最大峰值为独立峰值,其边界为该vort_min线
    for i in range(len(isolated_vort_min_record)):
        chosed_vort_min_line = isolated_vort_min_record.iloc[i]
        #vort_min_line线内峰值编号
        inline_index = chosed_vort_min_line.inline_index
        #判断vort_min_line线内独立峰值
        isolated_peak_num = np.sum(leaf.loc[inline_index,'is_isolated'])
        #更新leaf
        #当chosed_vort_min_line其中独立峰值为0时说明其中除了最大峰值无其他独立峰值，因此该vort_min_line分配给该峰值
        #当独立峰值数目为1时,该独立峰值必为最大峰值。反证法：如果某一的峰值为独立峰值，则其涡度必定满足强度判据，其邻接的涡度区域必定含有最大峰值，由于最大
        #峰值的涡度大于该峰值，因此最大峰值的涡度区域必定也独立，因此最大峰值的涡度区域必定存在一个独立峰值，则存在两个峰值，与前提不符。
        #当存在两个以上峰值时，说明存在两个以上涡度区域，不更新边界
        if isolated_peak_num<=1:
            leaf.loc[chosed_vort_min_line.max_peak_index,'vort_min_line'] = chosed_vort_min_line.vort_min_line
            leaf.loc[chosed_vort_min_line.max_peak_index,'vort_min'] = chosed_vort_min_line.vort_min
            leaf.loc[chosed_vort_min_line.max_peak_index,'is_isolated'] = True
    leaf['isolated_num'] = np.sum(leaf.is_isolated.values)
    return leaf
#4、用于判断一个或多个独立涡旋的真正边界以及面积，test_vort：一个大涡旋区域，其所有涡旋区域的original_vort_line相同
def area_deal(chosed_vort,chosed_original_vort_line,lon_lat_points,buffer=10):
    #计算独立涡旋的边界，以及质心
    #chosed_original_vort_line:选择的外边界线,chosed_vort：该边界线内的峰值
    #lon_lat_points：ERA5中的经纬度点(一维)
    #设置缓冲距离，当待分配的点与各个孤立涡旋边界的最小距离间距和第二小距离间距差距小于buffer；
    #-----------------------------------------------------------------------------
    #original_vort_line中包含点的bool
    out_vort_line_bool = chosed_original_vort_line.contains_points(lon_lat_points)
    if len(chosed_vort)==1:
       chosed_vort['isolated_area_path'] = [mpath.Path(chosed_original_vort_line.vertices)]
       return chosed_vort
   #----------------------------------------------------------------------------
    #以一个列表记录每个独立涡旋的vort_min线所包含的坐标点
    vort_min_inline_points_list = []       
    for main_vort_min_line in chosed_vort.vort_min_line.values:
        #vort_min_line包含的点的bool
        vort_min_point_bool = main_vort_min_line.contains_points(lon_lat_points)
        #记录每个独立涡旋的vort_min线所包含的坐标点
        vort_min_inline_points_list.append(lon_lat_points[vort_min_point_bool,:])
        #不断迭代找到original_vort_line内vort_min线外的点
        out_vort_line_bool = np.logical_and(out_vort_line_bool,~vort_min_point_bool)   
    #original_vort_line内vort_min线外的点
    out_points = lon_lat_points[out_vort_line_bool,:]
    #记录待分配的点与各个孤立涡旋极值点的距离
    out_line_points_distance = np.zeros((len(chosed_vort),len(out_points)))
   
    #认为难以区分，忽略这些点,否则可能出现直线相互重叠的边界
    #外循环是各个孤立涡旋边界，内循环是待分配的点
    for ii in range(len(chosed_vort)):
        for jj in range(len(out_points)):
            vort_min_line = chosed_vort.iloc[ii].vort_min_line.vertices
            out_line_points_distance[ii,jj] = geodistance(out_points[jj,0],out_points[jj,1],vort_min_line[:,0],vort_min_line[:,1]).min()
    #排序，以求最小距离间距和第二小距离间距差距
    sorted_distance = np.sort(out_line_points_distance,axis=0)
    distance_diff_bool = (sorted_distance[1,:]-sorted_distance[0,:])>buffer
    #-----------------------------------------------------------------
    #判断待分配的点与哪个独立涡旋最近,一维数组
    out_points_allocate = np.argmin(out_line_points_distance,axis=0)
    record_line_paths = []
    for i in range(len(chosed_vort)):
        #重分配的所有点
        all_vort_min_points = np.vstack([vort_min_inline_points_list[i],out_points[(out_points_allocate==i)&distance_diff_bool]])
        #生成凹包,从精确到模糊生成边界，如果无法生成则用原始边界
        alpha_flag = True
        for alpha in [3,2,1,0.5]:
            #alpha数字越大越精确，运行速度也会越慢 
            try:
                alpha_shape = alphashape.alphashape(all_vort_min_points,alpha)
                if alpha_shape.geom_type=='Polygon':
                    isolated_area_path = mpath.Path(np.array(alpha_shape.exterior.coords))
                    alpha_flag = False
                    break
            except:
                alpha_flag = True
            #print(alpha_shape.geom_type)
        if alpha_flag:
            isolated_area_path = mpath.Path(chosed_vort.iloc[i].vort_min_line.vertices)
        record_line_paths.append(isolated_area_path)   
    chosed_vort['isolated_area_path'] =  record_line_paths 
    return chosed_vort
#4、主程序
def distinguish_peaks(chosed_vort,parameter_sets,keeped_points=[]):
    smooth_r,vort_max0,useless_ax,vort_min0,scal_factor = parameter_sets
    lons,lats= chosed_vort.longitude,chosed_vort.latitude
    lon_grid,lat_grid = np.meshgrid(lons.values,lats.values,indexing = 'xy')
    lat_lon_points = np.vstack([lat_grid.reshape(-1),lon_grid.reshape(-1)]).T
    lat_lon_tree = BallTree(np.radians(lat_lon_points), leaf_size=20,metric = 'haversine')
    smoothed_data = smooth(chosed_vort,lat_lon_points,lat_lon_tree,smooth_r)
    #-------------------------------------------------------------------------------------------------------
    #threshold_rel :峰的最小强度，计算为 max(image) * threshold_rel 
    local_max_coordinate = peak_local_max(np.where(smoothed_data<vort_max0,0,smoothed_data),min_distance=4,exclude_border=True,threshold_rel=0.1)#找到大于vort_max0的极值点
    peak_values = smoothed_data[local_max_coordinate[:,0],local_max_coordinate[:,1]]
    peak_lons,peak_lats = lons[local_max_coordinate[:,1]],lats[local_max_coordinate[:,0]]
    peaks = pd.DataFrame(columns=(['lon','lat','peak_value']))#每一个极值都对应两个波谷和两个波谷线
    peaks['lon'],peaks['lat'],peaks['peak_value'] = peak_lons.values,peak_lats.values,peak_values
    if len(keeped_points):
        peaks = pd.merge(peaks,keeped_points.loc[:,['lon','lat']],how='inner').reset_index(drop=True)
    #涡度边界提取------------------------------------------------------------------------------------------------------------------------------
    vort_min0_lines = useless_ax.contour(lons,lats,smoothed_data,levels=[vort_min0],extend='both',colors = 'k',linestyles='--',linewidths=1,alpha=1, 
                                        transform =ccrs.PlateCarree()).collections[0].get_paths() #画边界
    #只选取等效半径大于100KM的涡旋，可以剔除掉格陵兰岛的常驻小涡旋，从而减少计算量
    diameter_threshold = 50
    line_area  = np.array([area({'type':'Polygon','coordinates':[line.vertices.tolist()]})/1e6 for line in vort_min0_lines])########   闭环,平方公里[(纬度，经度)]
    line_area_r = np.sqrt(line_area/np.pi)#等效半径为相同面积的圆的的半径
    original_lines = pd.DataFrame(vort_min0_lines,columns=['line_path'])
    original_lines['equal_r'] = line_area_r
    original_lines = original_lines.loc[original_lines['equal_r'] >diameter_threshold]
    #统计每个边界线中的峰值
    original_lines['inline_index'] = [peaks.loc[temp_line.contains_points(peaks.loc[:,['lon','lat']].values)].index.values for temp_line in original_lines.line_path] 
    #去掉没有峰值的边界线
    original_lines = original_lines.loc[[len(inline_index)>0 for inline_index in original_lines.inline_index] ]
    original_lines['error_border'] = False
    #判断极值
    record_leaf = []
    #生成独立峰值-------------------------------------------------------------------------------------
    for i in range(len(original_lines)):
        chosed_original_line = original_lines.iloc[i].copy()
        #等值线出现在边界时首尾连接
        if judge_frontier(chosed_original_line.line_path,lons,lats):
            points = chosed_original_line.line_path.vertices
            chosed_original_line.line_path = mpath.Path(np.vstack([points,points[0,:]]))
            #当线段两端的距离过长时，isolated_area_path可能会极大的错估，此时应记录下来并调整
            if geodistance(points[0,0],points[0,1],points[-1,0],points[-1,1])>800:
                #print(geodistance(points[0,0],points[0,1],points[-1,0],points[-1,1]))
                chosed_original_line.error_border = True
            original_lines.iloc[i] = chosed_original_line
        #生成峰值对应的等值线
        temp_peaks = peaks.loc[chosed_original_line.inline_index].copy()
        #如果峰值数目为1直接记录
        if len(temp_peaks)==1:
            leaf =  pd.DataFrame(data ={'vort_min':vort_min0,'vort_min_line':chosed_original_line.line_path,'original_vort_line':chosed_original_line.line_path,'is_isolated':True,'isolated_num':1},index = temp_peaks.index)
            record_leaf.append(pd.concat([temp_peaks,leaf],axis=1))
            continue
        #生成”分隔等值线“
        vort_min_record,leaf = find_vort_min(smoothed_data,chosed_original_line.line_path,temp_peaks,lons,lats,parameter_sets)
        #核心找独立涡度程序
        record_leaf.append(find_main_leaf(chosed_original_line,temp_peaks,vort_min_record,leaf,parameter_sets))
    if len(record_leaf)==0:
        return [None]*4
    all_main_leafs = pd.concat(record_leaf,axis=0)
    if len(keeped_points):
        all_main_leafs.is_isolated=True
    all_main_leafs = all_main_leafs.loc[all_main_leafs.is_isolated==True].drop('is_isolated',axis=1)
    return all_main_leafs,peaks,original_lines,smoothed_data

#5、融合引导风场------------------------------------------------------------------------------------------------------------
def wind_merge(u,v,pressure_levels,isolated_peaks,lons,lats): 
    #print(np.mean(chosed_u,axis=(0,1)).values,np.mean(chosed_v,axis=(0,1)).values)
    u,v = u.loc[pressure_levels].mean('level'),v.loc[pressure_levels].mean('level')
    r_smooth = 450#200KM内平滑风场,stoll(2021)
    steer_u,steer_v= [],[]
    for i in isolated_peaks.index:
        peak_lon_temp,peak_lat_temp = isolated_peaks.loc[i,['lon','lat']].values
        temp_bool = find_r_points(r_smooth,peak_lon_temp,peak_lat_temp,lons,lats)
        steer_u.append(u.values[temp_bool].mean())
        steer_v.append(v.values[temp_bool].mean())
    isolated_peaks['steer_u'] = steer_u
    isolated_peaks['steer_v'] = steer_v
    isolated_peaks['next_lon'],isolated_peaks['next_lat'] = geo2lonlat_uv(isolated_peaks.lon.astype('float'),isolated_peaks.lat.astype('float'),
                                                                          isolated_peaks.steer_u.astype('float'),isolated_peaks.steer_v.astype('float'))
    isolated_peaks = isolated_peaks.reset_index(drop=True)#编号重排
    return isolated_peaks

#二、涡旋追踪
##################################################################################################################################################################
#寻找匹配的涡旋
def find_next_vor_r(isolated_peaks1,isolated_peaks2):#这里使用r km 的圆来检测
    #isolated_peaks1是当前时次的某个isolated_peak,格式为一行的series,注意是一行
    #isolated_peaks2是下一时次找到的所有isolated_peak,函数的目的是为了匹配连续时间步的peak
    next_lon,next_lat = isolated_peaks1[['next_lon','next_lat']].values
    r = 180
    target_peak = None
    #值得注意的是也可以使用半径为150km的圆（stoll,2021）,经检验还是用180吧
    #第一步，找到预测位置范围内的峰值--------------------------------------------------------------------------------------------------------
    find_bool1 = geodistance(next_lon,next_lat,isolated_peaks2.lon.values,isolated_peaks2.lat.values)<r
    if sum(find_bool1)==1:
        target_peak = isolated_peaks2[find_bool1].squeeze()#统一为series格式
    #若峰值数量大于1，找到最近的峰值
    elif sum(find_bool1)>1:
        candidate_peak = isolated_peaks2.loc[find_bool1,:]
        nearest = np.argmin(geodistance(next_lon,next_lat,candidate_peak.lon.values,candidate_peak.lat.values))
        target_peak = candidate_peak.iloc[nearest,:]
    #若未找到峰值，需要判断每个峰值的vort_min区域的重叠程度,找到重叠程度最大的
    #需不需要包括多峰值区域呢
    elif sum(find_bool1)==0:
        #先构建一个预测位置方形区域的格点，方便选圆形内的点
        temp_lon_grid,temp_lat_grid = np.meshgrid(np.arange(next_lon-10,next_lon+10,0.1),
                                    np.arange(next_lat-10,next_lat+10,0.1),indexing = 'xy')
        temp_lon_lat_points = np.vstack([temp_lon_grid.reshape(-1),temp_lat_grid.reshape(-1)]).T
        #再提取150KM内的格点，判断这些格点在vort_min线内的个数
        temp_lon_lat_points = temp_lon_lat_points[geodistance(next_lon,next_lat,temp_lon_lat_points[:,0],temp_lon_lat_points[:,1])<r]
        overlap_degrees = []
        for temp_vort_min_line in isolated_peaks2.isolated_area_path:
            overlap_degrees.append(sum(temp_vort_min_line.contains_points(temp_lon_lat_points)))
        #重叠的点大于15个即可
        if max(overlap_degrees)>15:
            target_peak = isolated_peaks2.iloc[np.argmax(overlap_degrees)]

    if type(target_peak) is not type(None):
        return target_peak,geodistance(next_lon,next_lat,target_peak.lon,target_peak.lat)#多返回一个距离，到时候用来判断多对一的情况
    else:
        return None,None
#存储涡旋的时间步及连接信息------------------------------------------------------------------------------------------------------------------------------------
#定义结点，每个结点应有基本的信息：记录的dataframe
class Node():		
    def __init__(self,value,next=None):		
        self.value = value					
        self.next = next
#定义链表，self.next总是指向最后一个结点,root:根节点
class LinkedList(object):  			
    def __init__(self,root=None,start=True):	
        #start记录此路径的开始是因为文件的开始还是新发现的vortex
        self.root = Node(root)			
        self.size = 1				
        self.next = None
        self.start=start 			
    def add(self,value):
        node = Node(value)
        #判断是否已经有数据
        if not self.next:
            self.root.next = node #将新节点挂到root后
        else:
            self.next.next = node
        self.next = node
        self.size+=1
    #返回路径最后的vort
    def return_leaf(self):
        if self.next:
            return self.next.value
        else:
            return self.root.value
    def return_all_node(self):
        record_node = []
        next_node = self.root
        while next_node:
            record_node.append(next_node.value)
            next_node = next_node.next
        return record_node
#一轮匹配
def round_track(running_record,isolated_peaks_next):
    if len(isolated_peaks_next)==0:
        return [],[],[]
      #closed_record记录已经消亡的vortex
    #running_record记录仍存在的vortex
    temp_ruuning_record = []
    finded_vort_current = []
    current_record = {}
    #finded_vort_current记录当前时刻找到的和上一时刻匹配的涡旋
    #对上一时刻的涡旋进行追踪
    for path in running_record:
        target_vort,target_distance = find_next_vor_r(path.return_leaf(),isolated_peaks_next)
        if target_vort is not None:
            if target_vort.name not in finded_vort_current:
                #第一次发现，存入字典，这里的name是series的序号
                current_record[target_vort.name] = [(path,target_distance)]
            else:
                current_record[target_vort.name].append((path,target_distance))
            finded_vort_current.append(target_vort.name)#注意series要用name而非index
    #在这里对多对一的涡旋进行处理，实际上一对一的也包括在内
    dropped_path = []
    #记录isolated_peaks_next中所有被找到的编号
    for target_vort_index, v in current_record.items():
        sorted_items = sorted(v,key=lambda x:x[1])
        path,_ = sorted_items[0]#按照距离排序,找到最近的
        #除了最近的path,剩下的path进一步记录并判断
        if len(sorted_items)>1:
           dropped_path +=  [i[0] for i in sorted_items[1:]]
        #将新节点挂到root后
        path.add(isolated_peaks_next.loc[target_vort_index])
        temp_ruuning_record.append(path)
    return dropped_path,finded_vort_current,temp_ruuning_record
#涡旋追踪，追踪的是isolated_peaks的一行------------------------------------------------------------------------------------------------------------------------
def track_vort(running_record,closed_record,isolated_peaks_next):
    #第一次匹配
    dropped_path,finded_vort_current,temp_ruuning_record = round_track(running_record,isolated_peaks_next)
    #-------------------------------------------------------------------------------------
    #第二轮匹配
    if len(dropped_path):
        other_isolated_peaks = isolated_peaks_next.drop(finded_vort_current,axis=0)
        _,new_finded_vort_current,new_temp_ruuning_record = round_track(dropped_path,other_isolated_peaks)
        temp_ruuning_record = temp_ruuning_record+new_temp_ruuning_record
        finded_vort_current = finded_vort_current+new_finded_vort_current
    #没有发现的将被存入closed_record中
    temp_closed = [path for path in  running_record if path not in temp_ruuning_record]
    #同时temp_ruuning_record增添在这一时刻新增的涡旋，为新增的路径，start定义为True---------------------------------------------
    for index, row in isolated_peaks_next.iterrows():
        if index  not in finded_vort_current:
            temp_ruuning_record.append(LinkedList(root=row,start=True))
    return temp_ruuning_record,closed_record+temp_closed

#改写为两个时刻涡旋的连接---------------------------------------------------------------------
def round_connect(isolated_peaks_first,isolated_peaks_next):
    if len(isolated_peaks_next)==0:
        return [],[],[]
    #finded_vort_current记录当前时刻找到的和上一时刻匹配的涡旋
    finded_next_indexs,current_record = [],{}
    #对上一时刻的涡旋进行追踪
    for _,path in isolated_peaks_first.iterrows():
        target_vort,target_distance = find_next_vor_r(path,isolated_peaks_next)
        if target_vort is not None:
            if target_vort.name not in finded_next_indexs:
                #第一次发现，存入字典，这里的name是series的序号
                current_record[target_vort.name] = [(path,target_distance)]
            else:
                current_record[target_vort.name].append((path,target_distance))
            finded_next_indexs.append(target_vort.name)#注意series要用name而非index
    #在这里对多对一的涡旋进行处理，实际上一对一的也包括在内
    dropped_paths,findded = [],[]
    #记录isolated_peaks_next中所有被找到的编号
    for target_vort_index, v in current_record.items():
        sorted_items = sorted(v,key=lambda x:x[1])
        path,_ = sorted_items[0]#按照距离排序,找到最近的
        #除了最近的path,剩下的path进一步记录并判断
        if len(sorted_items)>1:
           dropped_paths +=  [i[0] for i in sorted_items[1:]]
        #记录连接轨迹
        findded.append([path.path_index,isolated_peaks_next.loc[target_vort_index,'path_index']])
    if len(dropped_paths):
        dropped_paths = pd.concat([i.to_frame().T for i in dropped_paths],axis=0)
    return dropped_paths,finded_next_indexs,findded
#涡旋追踪，追踪的是isolated_peaks的一行------------------------------------------------------------------------------------------------------------------------
def track_connect(isolated_peaks_first,isolated_peaks_next):
    #第一次匹配
    dropped_paths,finded_next_indexs,findded = round_connect(isolated_peaks_first,isolated_peaks_next)
    new_finded = []
    #-------------------------------------------------------------------------------------
    #第二轮匹配，是为了给第一次因最近邻匹配剩下的涡旋再进行一次匹配
    if len(dropped_paths):
        other_isolated_peaks = isolated_peaks_next.drop(finded_next_indexs)
        if len(other_isolated_peaks):
            _,_,new_finded = round_connect(dropped_paths,other_isolated_peaks)
    return np.array(findded+new_finded)



