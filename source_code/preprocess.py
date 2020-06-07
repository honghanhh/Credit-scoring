import pandas as pd
import numpy as np
import re
import os

class Preprocess():
    def __init__(self, df):
        self.df = df

    #Age range: 18 - 65
    def combined_age(self):
        self.df['age'] = self.df['age_source2'].mask(pd.isnull, self.df['age_source1'])
        self.df['age'][self.df['age']<18] = self.df['age'].median()
        self.df['age'][self.df['age']>65] = self.df['age'].median()
        self.df.drop(["age_source1", "age_source2"], axis = 1, inplace = True)
        return self.df


    def data_normalization(self):
        #Handle mixing values
        self.df['FIELD_9'].replace(['75','74','80','86','79'],np.nan,inplace = True)
        self.df["FIELD_12"].replace(['DN','GD','TN','HT','DN','XK','DT','DK'],np.nan,inplace = True)
        self.df["FIELD_13"].replace(['0','4','8'],np.nan,inplace = True)
        self.df["FIELD_39"].replace(['1'],np.nan,inplace = True)
        self.df["FIELD_40"].replace(['02 05 08 11','05 08 11 02','08 02'],np.nan,inplace = True)
        self.df["FIELD_43"].replace(['0','5'],np.nan,inplace = True)
        #Standardize NAN, T, F formats
        map_true = ['True', 'TRUE']
        map_false = ['False', 'FALSE']
        map_nan = ['NaN','nan','na']
        self.df.replace(map_true, True, inplace = True)
        self.df.replace(map_false, False, inplace = True)
        self.df.replace(map_nan, np.nan,inplace = True)
        self.df.replace('None',-1, inplace=True)
        return self.df 

    def overdue(self):
        self.df['FIELD_3_365'] = (self.df['FIELD_3']/365).round(0) #number of years
        self.df['FIELD_3_RESIDUAL'] = self.df['FIELD_3_365']*365 - self.df['FIELD_3'] #overdue days
        self.df.drop('FIELD_3_365', axis=1, inplace = True)
        return self.df

    def map_value(self, list_, text, str_):
        for i in range(len(list_)):
            if list_[i] in text:
                text = str_
        return text
    
    
    def normalize_province(self):
        self.df['region'] = [str(x).lower() for x in self.df['province']]
        self.df['region'] = [re.sub(' +',' ',x) for x in self.df['region']]
        # row = pd.Series(row)
        self.df['region'].replace('tỉnh hòa bình', 'tỉnh hoà bình', inplace=True)
        self.df['region'].replace('tỉnh vĩnh phúc', 'tỉnh vĩnh phúc', inplace=True)
        bac_bo = ['hoà bình','sơn la' ,'điện biên','lai châu' ,'lào cai','yên bái']
        dong_bac_bo = ['quảng ninh' , 'phú thọ','hà giang' ,'tuyên quang', 'cao bằng', 'bắc kạn','thái nguyên' ,'lạng sơn' ,'bắc giang'] 
        song_hong = ['hà nội', 'bắc ninh', 'hà nam' , 'hải dương' , 'hải phòng','hưng yên','nam định','thái bình','vĩnh phúc','ninh bình']
        bac_trung_bo = ['thanh hóa','nghệ an','hà tĩnh' ,'quảng bình','quảng trị' ,'thừa thiên huế' ]
        nam_trung_bo = ['khánh hòa','đà nẵng' ,'quảng nam','quảng ngãi','bình định' ,'phú yên' ,'ninh thuận' ,'bình thuận' ] 
        tay_nguyen = ['kon tum','gia lai','đắk lắk','đắk nông' , 'lâm đồng'] 
        dong_nam_bo = ['hồ chí minh' ,'vũng tàu' ,'bình dương','bình phước' ,'đồng nai' ,'tây ninh']
        cuu_long = ['an giang' ,'bạc liêu' ,'bến tre' ,'cà mau', 'cần thơ','đồng tháp' ,'hậu giang' , 'kiên giang' ,'long an' ,'sóc trăng' ,'tiền giang' ,'trà vinh' ,'vĩnh long'] 

        for i in range(len(self.df['region'])):
            self.df['region'].iloc[i] = self.map_value(bac_bo,self.df['region'].iloc[i],'bắc bộ')
            self.df['region'].iloc[i] = self.map_value(dong_bac_bo,self.df['region'].iloc[i],'đông bắc bộ')
            self.df['region'].iloc[i] = self.map_value(song_hong,self.df['region'].iloc[i],'đồng bằng sông hồng')
            self.df['region'].iloc[i] = self.map_value(bac_trung_bo,self.df['region'].iloc[i],'bắc trung bộ')
            self.df['region'].iloc[i] = self.map_value(nam_trung_bo,self.df['region'].iloc[i],'nam trung bộ')
            self.df['region'].iloc[i] = self.map_value(tay_nguyen,self.df['region'].iloc[i],'tây nguyên')
            self.df['region'].iloc[i] = self.map_value(dong_nam_bo,self.df['region'].iloc[i],'đông nam bộ')
            self.df['region'].iloc[i] = self.map_value(cuu_long,self.df['region'].iloc[i],'đồng bằng sông cửu long')
        return self.df  

    def normalize_maCv(self):
        self.df['job_cluster'] = [str(x).lower() for x in self.df['maCv']]
        self.df['job_cluster'] = [re.sub(' +',' ',x) for x in self.df['job_cluster']]
        labor = ['gò dán','cạo mủ','gia công','lái máy','ép suất','gia công','khiêng khuôn','thêu vi tính','lái','kiểm hàng','bếp chính','mài','trải vải',
                'khai thác', 'ép da','ép đế','thuyền viên','đứng máy', 'sửa chữa','đầu bếp','ép kim','mài da', 'sàng sấy','đổ sợi' ,'hàn điện','mộc máy',
                'bảo trì', 'cắt bo','vệ sinh','lao công','bôi keo','phết keo','đứng bếp','may','giao hàng','lắp ráp','lao công','phụ may','lao động phổ thông',
                'đóng gói','phụ','scan mẫu','bóc xếp','lỏi xe','đóng đinh','phụ lái','giao nhận','pha chế','kiểm giày','tap vu','phụ xế','b.vệ','b. vệ',
                'cắt cỏ','phụ việc','tài xê','thủ công','phụ xe','công nhân','cộng nhân','côn gnhân','coõng nhaõn','cn', 'tạp vụ','cnhân','công phụ','thợ may',
                'đứng máy cán','phục vụ','c.n','bảo vệ','thợ','ldpt','lđ','làm mác','nấu ăn','cụng nhõn','tài xế','thời vụ','lao động','bốc xếp','cắt vải']
        medical_edu = ['dược tá', 'dược sĩ','dựoc sỹ','kiểm diịch viên','trị liệu','bác sỹ','bác sĩ','y sỹ','y tá','y sĩ','hộ sinh','hộ lý','điều dưỡng',
                        'y tế','dược sỹ','dược viên','bệnh viện','trạm y tế','bs.','khoa','giáo viên','giaó viên','gíao viên','đào tạo','tổ phó','chủ nhiệm',
                        'bảo mẫu','giảng viên','bộ môn','hiệu trưởng','hiệu phó','gv','giỏo viờn']
        govern = ['biên chế', 'địa chính', 'cb', 'tư pháp', 'trạm phó', 'ủy viên','kiểm tra viên','thanh tra','uỷ viên','đảng','uỷ nhiệm', 'công chức',
                  'cán sự', 'viên chức', 'đảng uỷ', 'chỉ huy', 'chánh văn phòng', 'uỷ viên', 'sĩ quan', 'bí tư','bí thư', 'cán bộ', 'đoàn xã', 'phường','đảng uỷ',
                  'bí thư','mặt trận','tổ quốc','quân sự','tổ trưởng','hội liên hiệp phụ nữ','đoàn thanh niên','công an', 'ban dân vận','ubmttq', 'xã đội'] 
        staff = ['phóng viên','văn phòng','gdv quỹ' ,'mậu dịch viên', 'kiểm tra chất lượng', 'thao tác viên' ,'biên kịch' ,'kiểm hoá','thu? kho','sản xuất','thủy thủ',
                 'k? thu?t viờn','kế tóan' ,'kỷ thuật','quan trắc viện','kiểm ngân viên','kiểm ngân viên' ,'quản trị viên' ,'kiểm hoá' ,'tổng đài' ,'gia?m sa?t',
                 'thủ quỹ','quan trắc viên' ,'thu ngân','kiến trúc sư','lập trình','giám định viên','chế tác viên' ,'huấn luyện viên','kiểm soát viên',
                 'nhan vien','diễn viên','tiếp tân','văn thư','thư ký','tư vấn viên','kiểm lâm','kinh doanh','bán hàng','thiết kế','vận hành','thủ kho',
                  'kỹ thuật','sale assistant' ,'tiếp thị','tiếp viên','thư kí' ,'kiểm lâm','tư vấn','nhaõn vieõn' ,'nhâ viên' ,'n.viên','nhõn viờn' ,'nhân viên',
                  'nv','chuyên viên' ,'kĩ thuật viên','ktv' ,'kỹ sư' ,'trợ lý','kỹ thuật viên','kế toán' ,'kỹ sư' ,'giao dịch viên' ,'hướng dẫn viên']
        manager = ['phóng viên', 'nhà báo', 'đài truyền thanh', 'p.gđốc','cửa hàng trưởng' ,'manager' ,'ql' ,'trưởng' ,'đội phó' ,'qu?n lý','đội trưởng',
                    'trưởng bộ phận','trưởng nhóm','trưởng phòng','đại diện','cửa hàng trưởng','giám đốc' ,'phó phòng' ,'chủ tịch' ,'quản lí' ,'quản lý','executive',
                    'tổ trưởng', 'quản lý', 'giám sát', 'điều hành' ,'đội trưởng','quản đốc' ,'chủ quản']
        for i in range(len(self.df['job_cluster'])):
            self.df['job_cluster'].iloc[i] = self.map_value(labor,self.df['job_cluster'].iloc[i],'lao động phổ thông')
            self.df['job_cluster'].iloc[i] = self.map_value(medical_edu,self.df['job_cluster'].iloc[i],'y tế giáo dục')
            self.df['job_cluster'].iloc[i] = self.map_value(staff,self.df['job_cluster'].iloc[i],'nhân viên')
            self.df['job_cluster'].iloc[i] = self.map_value(govern,self.df['job_cluster'].iloc[i],'nhà nước')
            self.df['job_cluster'].iloc[i] = self.map_value(manager,self.df['job_cluster'].iloc[i],'quản lý')
            # print(self.df['job_cluster'].iloc[i])
        count = self.df['job_cluster'].value_counts()
        minority = self.df['job_cluster'].isin(count.index[count < 5])
        self.df.loc[minority,'job_cluster'] = np.NaN
        return self.df 

    def generate_statistic(self):
        short_columns = dict([(f'FIELD_{i}', str(i)) for i in range(1, 58)])
        ignore_columns = '36 37 label id'.split()
        self.df.rename(columns=short_columns, inplace=True)
        columns = set(self.df.columns).difference(ignore_columns)
        self.df['count_NaN'] = self.df[columns].isna().sum(axis=1)
        self.df['count_True'] = self.df[columns].applymap(lambda x: isinstance(x, bool) and x).sum(axis=1)
        self.df['count_False'] = self.df[columns].applymap(lambda x: isinstance(x, bool) and not x).sum(axis=1)
        return self.df

    def preprocess(self):
        self.df = self.combined_age()
        self.df = self.data_normalization()
        self.df = self.overdue()
        self.df = self.generate_statistic()
        self.df = self.normalize_province()
        self.df = self.normalize_maCv()
        return self.df

if __name__ == "__main__":
    train_pd = pd.read_csv('../data/train.csv')
    preprocess = Preprocess(train_pd)
    df = preprocess.preprocess()
    df.to_csv('../data/clean_train.csv',index =False)
