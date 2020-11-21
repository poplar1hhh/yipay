# coding: utf-8
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from sklearn.model_selection import KFold
import gc
from sklearn import preprocessing
from scipy.stats import mode
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
import datetime
time=datetime.date.today()
time=str(time)[-5:]
print(time)
warnings.filterwarnings('ignore')


def load_dataset(DATA_PATH):
    train_label = pd.read_csv(DATA_PATH+'train_label.csv')
    train_base = pd.read_csv(DATA_PATH+'train_base.csv')
    test_base = pd.read_csv(DATA_PATH+'testb_base.csv')

    train_op = pd.read_csv(DATA_PATH+'train_op.csv')
    train_trans = pd.read_csv(DATA_PATH+'train_trans.csv')
    test_op = pd.read_csv(DATA_PATH+'testb_op.csv')
    test_trans = pd.read_csv(DATA_PATH+'testb_trans.csv')
    train_trans = train_trans.drop_duplicates()
    return train_label, train_base, test_base, train_op, train_trans, test_op, test_trans

#处理时间
def transform_time(x):
    day = int(x.split(' ')[0])
    hour = int(x.split(' ')[2].split('.')[0].split(':')[0])
    minute = int(x.split(' ')[2].split('.')[0].split(':')[1])
    second = int(x.split(' ')[2].split('.')[0].split(':')[2])
    return 86400*day+3600*hour+60*minute+second

def labelEncoder_df(df,features):
    for i in features:
        encoder = preprocessing.LabelEncoder()
        df[i] = encoder.fit_transform(df[i])

#预处理
def data_preprocess(DATA_PATH):
    train_label, train_base, test_base, train_op, train_trans, test_op, test_trans = load_dataset(DATA_PATH=DATA_PATH)
    # 拼接数据
    train_df = train_base.copy()
    test_df = test_base.copy()
    train_df = train_label.merge(train_df, on=['user'], how='left')
    del train_base, test_base

    op_df = pd.concat([train_op, test_op], axis=0, ignore_index=True)
    trans_df = pd.concat([train_trans, test_trans], axis=0, ignore_index=True)
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)



    # 缺失值处理
    cols = ['sex', 'balance_avg', 'balance1_avg', 'provider', 'province', 'city','level']
    for col in cols:
        data[col].fillna(r'\N', inplace=True)
    cols = ['balance_avg','balance1_avg','level']
    for col in cols:
        data[col].replace({r'\N': -1}, inplace=True)
        data[col] = data[col]

    # 转等级和连续型
    cols_int = [f for f in data.columns if
                f in ['level', 'balance', 'balance_avg', 'balance1', 'balance1_avg', 'balance2', 'balance2_avg',
                      'product1_amount', 'product2_amount', 'product3_amount', 'product4_amount', 'product5_amount',
                      'product6_amount']]
    for col in cols_int:
        for i in range(0, 50):
            data.loc[data[col] == "category %d" % i, col] = i
            data.loc[data[col] == "level %d" % i, col] = i
        data[col].isnull().sum()
        data[col].astype(int)


    #转类型为字符型编码
    cols = ['sex','provider','verified','regist_type','agreement1','agreement2','agreement3','agreement4','province','city','service3']
    for col in cols:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str)
            labelEncoder_df(data, cols)
    print(data.info())

    del train_op, test_op, train_df, test_df
    # 时间维度的处理
    op_df['days_diff'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[0]))
    trans_df['days_diff'] = trans_df['tm_diff'].apply(lambda x: int(x.split(' ')[0]))
    op_df['timestamp'] = op_df['tm_diff'].apply(lambda x: transform_time(x))
    trans_df['timestamp'] = trans_df['tm_diff'].apply(lambda x: transform_time(x))
    op_df['hour'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[0]))
    trans_df['hour'] = trans_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[0]))
    trans_df['week'] = trans_df['days_diff'].apply(lambda x: x % 7)
    # 排序
    trans_df = trans_df.sort_values(by=['user', 'timestamp'])
    op_df = op_df.sort_values(by=['user', 'timestamp'])
    trans_df.reset_index(inplace=True, drop=True)
    op_df.reset_index(inplace=True, drop=True)

    gc.collect()#gc.collect()清内存
    return data, op_df, trans_df


def gen_user_amount_features(df):
    stat_list = ['min', 'max', 'mean', 'std', 'median', 'skew', 'var', 'sum','count']
    group_df = df.groupby(['user'])['amount'].agg(stat_list).reset_index()
    print(group_df.shape,list(group_df.columns))
    group_df.columns=['user']+['user_amount_'+str(n) for n in ['min', 'max', 'mean', 'std', 'median', 'skew', 'var','sum','cnt']]
    return group_df

def gen_user_count_features(df,value):
    group_df = df.groupby(['user'])[value].agg({
        'user_{}_count'.format(value): 'count'
        }).reset_index()
    return group_df

def intersection_op_trans(op,trans):#交集
    return len(op & trans)

def union_op_trans(op,trans):#并集
    return len(op | trans)

def difference_op_trans(op,trans):
    return len(op - trans)

def difference_trans_op(op,trans):
    return len(trans - op)

def symmetric_trans_op(op,trans):
    return len(op ^ trans)

def gen_user_group_amount_features(df, value):
    group_df = df.pivot_table(index='user',
                              columns=value,#platform
                              values='amount',
                              dropna=False,
                              aggfunc=['count', 'sum','max','min']).fillna(0)
    group_df.columns = ['user_{}_{}_amount_{}'.format(value, f[1], f[0]) for f in group_df.columns]
    group_df.reset_index(inplace=True)
    return group_df

def gen_user_group_timestamp_features(df, value):
    group_df = df.pivot_table(index='user',
                              columns=value,#platform
                              values='timestamp',
                              dropna=False,
                              aggfunc=['max']).fillna(0)
    group_df.columns = ['user_{}_{}_timestamp_{}'.format(value, f[1], f[0]) for f in group_df.columns]
    group_df.reset_index(inplace=True)

    return group_df


def op_do(data):
    aa=data.groupby(by=['user'],as_index=False)['op_type'].agg({'op_count':'count'})   #总的点击次数
    op_type_values=data.op_type.unique().tolist()     #page_no的列取值
    op_type_values.sort()          #排序

    a1=data.groupby(['user','days_diff'])['op_type'].count().unstack().reset_index()          #单日最大点击次数
    col=a1.columns.tolist()[1:]
    a1['click_day_max']=a1[col].max(axis=1)
    a1=a1[['user','click_day_max']]
    ##
    a2=data.groupby(by=['user'], as_index=False)['days_diff'].agg({'click_day_num': 'nunique'})   #点击天数
    aa=pd.merge(aa,a1,on='user',how='left')
    aa=pd.merge(aa,a2,on='user',how='left')
    aa['click_day_count']=aa['op_count']/aa['click_day_num']
    aa.drop(['click_day_num'],axis=1,inplace=True)

    return aa

def gen_user_window_amount_features(df, window):
    group_df = df[df['days_diff']>window].groupby('user')['amount'].agg({
        'user_amount_mean_{}d'.format(window): 'mean',
        'user_amount_std_{}d'.format(window): 'std',
        'user_amount_max_{}d'.format(window): 'max',
        'user_amount_min_{}d'.format(window): 'min',
        'user_amount_sum_{}d'.format(window): 'sum',
        'user_amount_med_{}d'.format(window): 'median',
        'user_amount_cnt_{}d'.format(window): 'count',
        }).reset_index()
    return group_df
def gen_user_window_amount_hour_features(df, window):
    group_df = df[df['hour']>window].groupby('user')['amount'].agg({
        'user_amount_mean_{}h'.format(window): 'mean',
        'user_amount_std_{}h'.format(window): 'std',
        'user_amount_max_{}h'.format(window): 'max',
        'user_amount_min_{}h'.format(window): 'min',
        'user_amount_sum_{}h'.format(window): 'sum',
        'user_amount_med_{}h'.format(window): 'median',
        'user_amount_cnt_{}h'.format(window): 'count',
        }).reset_index()
    return group_df
def gen_user_window_amount_week_features(df, window):
    group_df = df[df['week']==window].groupby('user')['amount'].agg({
        'user_amount_mean_{}w'.format(window): 'mean',
        'user_amount_std_{}w'.format(window): 'std',
        'user_amount_max_{}w'.format(window):'max',
        'user_amount_min_{}w'.format(window): 'min',
        'user_amount_sum_{}w'.format(window):'sum',
        'user_amount_med_{}w'.format(window):'median',
        'user_amount_cnt_{}w'.format(window):'count',
        }).reset_index()
    return group_df


def gen_user_nunique_features(df, value, prefix):
    group_df = df.groupby(['user'])[value].agg({
        'user_{}_{}_nuniq'.format(prefix, value): 'nunique'
    }).reset_index()
    return group_df

def gen_user_null_features(df, value, prefix):
    df['is_null'] = 0
    df.loc[df[value].isnull(), 'is_null'] = 1

    group_df = df.groupby(['user'])['is_null'].agg({'user_{}_{}_null_cnt'.format(prefix, value): 'sum',
                                                    'user_{}_{}_null_ratio'.format(prefix, value): 'mean'}).reset_index()
    return group_df


def gen_user_tfidf_features(df, value):
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()#把每个用户的op_mode转成列表
    group_df.columns = ['user', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))#将op_mode用，连接
    enc_vec = TfidfVectorizer()#得到tf-idf矩阵
    tfidf_vec = enc_vec.fit_transform(group_df['list'])#得到词频矩阵，将op_mode转为词向量，即计算机能识别的编码
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2020)#降维，提取op_mode的特征，TtuncatedSVD和SVD:TSVD可以选择需要提取的维度
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_tfidf_{}_{}'.format(value, i) for i in range(10)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df


def gen_user_countvec_features(df, value):
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    group_df.columns = ['user', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_vec = CountVectorizer()
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2020)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_countvec_{}_{}'.format(value, i) for i in range(10)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df

def kfold_stats_feature(train, test, feats, k):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=44)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = ['label']
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_trn = train.iloc[trn_idx]
                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = train[f].mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            test[colname] = None
            order_label = train.groupby([feat])[f].mean()
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = train[f].mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test

def gen_features(df, op, trans):
    cols = ['platform','tunnel_in','tunnel_out','type1','type2']
    for col in cols:
        trans[col].fillna('-999', inplace=True)


    df.drop(['service3_level'], axis=1, inplace=True)
    # base
    df['product7_fail_ratio'] = df['product7_fail_cnt'] / df['product7_cnt']
    df['city_count'] = df.groupby(['city'])['user'].transform('count')
    df['province_count'] = df.groupby(['province'])['user'].transform('count')
    df['using_time_service3'] = df['using_time']* df['service3']
    df['using_time_level'] = df['using_time'] * df['level']
    df['login_cnt_peridod_ratio'] = df['login_cnt_period1'] / df['login_cnt_period2']
    df['service3_balance_avg'] = df['service3'] * df['balance_avg']
    df['product7_cnt * product7_fail_cnt'] = df['product7_fail_cnt'] * df['product7_cnt']
    df['sex*using_time'] = df['using_time']* df['sex']
    df['product7_cnt*2']=df['product7_cnt'] * df['product7_cnt']
    df['using_time*agreement_total'] = df['using_time'] * df['agreement_total']
    df['using_time_age'] = df['using_time'] * df['age']
    df['age*acc_count'] = df['age']*df['acc_count']
    df['sex*age'] = df['age'] * df['sex']
    df['using_time * regist_type'] = df['regist_type'] *df['using_time']
    df['using_time * card_a_cnt'] = df['using_time'] * df['card_a_cnt']
    df['using_time * op1_cnt'] = df['using_time'] * df['op1_cnt']
    df['agreement_total * login_days_cnt'] = df['agreement_total'] * df['login_days_cnt']
    df['age * op1_cnt'] = df['age'] * df['op1_cnt']
    df['login_cnt_period1 * product6_amount'] = df['login_cnt_period1'] * df['product6_amount']
    df['login_cnt_avg * sex'] = df['login_cnt_avg'] * df['sex']
    df['login_cnt_avg * ip_cnt'] = df['login_cnt_avg'] * df['ip_cnt']
    df['age * agreement3'] = df['age'] * df['agreement3']

    # trans
    df = df.merge(gen_user_amount_features(trans), on=['user'], how='left')

    for col in tqdm(['days_diff', 'platform', 'tunnel_in', 'tunnel_out', 'type1', 'type2', 'ip', 'ip_3']):
        df = df.merge(gen_user_nunique_features(df=trans, value=col, prefix='trans'), on=['user'], how='left')
    df['user_amount_per_days'] = df['user_amount_sum'] / df['user_trans_days_diff_nuniq']
    df['user_amount_per_cnt'] = df['user_amount_sum'] / df['user_amount_cnt']
    df['op1_user_amount_cnt'] =df['op1_cnt'] / df['user_amount_cnt']
    df['op2_user_amount_cnt'] = df['op2_cnt'] / df['user_amount_cnt']

    df = df.merge(gen_user_group_amount_features(df=trans, value='platform'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(df=trans, value='type1'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(df=trans, value='type2'), on=['user'], how='left')

    df = df.merge(op_do(data=op), on=['user'], how='left')
    df['CTR'] = df['op_count'] / df['user_amount_cnt']


    df['using_time_acc_count_ritio'] = df['using_time'] / df['acc_count']
    df['user_amount_cnt_acc_count'] = df['user_amount_cnt']/df['acc_count']
    df['agreement_total_acc_count'] = df['agreement_total'] / df['acc_count']
    df['login_cnt_period1_acc_count'] = df['login_cnt_period1'] / df['acc_count']
    df['login_cnt_period2_acc_count'] = df['login_cnt_period2'] / df['acc_count']
    df['ip_cnt_acc_count'] = df['ip_cnt'] / df['acc_count']
    df['login_cnt_avg_acc_count'] = df['login_cnt_avg'] / df['acc_count']
    df['login_days_cnt_acc_count'] = df['login_days_cnt'] / df['acc_count']
    df['product7_cnt_acc_count'] = df['product7_cnt'] / df['acc_count']
    df['product7_fail_cnt_acc_count'] = df['product7_fail_cnt'] / df['acc_count']
    df['op1_cnt_op2_cnt_ritio'] = df['op1_cnt'] / df['op2_cnt']

    df = df.merge(gen_user_group_amount_features(df=trans, value='tunnel_out'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(df=trans, value='tunnel_in'), on=['user'], how='left')
    df = df.merge(gen_user_group_timestamp_features(df=trans, value='type1'), on=['user'], how='left')
    df = df.merge(gen_user_group_timestamp_features(df=trans, value='type2'), on=['user'], how='left')


    df = df.merge(gen_user_window_amount_features(df=trans, window=15), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=10), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=7), on=['user'], how='left')

    df = df.merge(gen_user_window_amount_hour_features(df=trans, window=0), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_hour_features(df=trans, window=6), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_hour_features(df=trans, window=10), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_hour_features(df=trans, window=12), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_hour_features(df=trans, window=18), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_hour_features(df=trans, window=24), on=['user'], how='left')

    df = df.merge(gen_user_window_amount_week_features(df=trans, window=1), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_week_features(df=trans, window=2), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_week_features(df=trans, window=3), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_week_features(df=trans, window=4), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_week_features(df=trans, window=5), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_week_features(df=trans, window=6), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_week_features(df=trans, window=0), on=['user'], how='left')



    df = df.merge(gen_user_null_features(df=trans, value='ip', prefix='trans'), on=['user'], how='left')
    group_df = trans[trans['type1']=='45a1168437c708ff'].groupby(['user'])['days_diff'].agg({'user_type1_45a1168437c708ff_min_day': 'min'}).reset_index()
    df = df.merge(group_df, on=['user'], how='left')
    group_df = trans.groupby(['user'])['timestamp'].agg({'user_max_time':'max'}).reset_index()
    df = df.merge(group_df, on=['user'], how='left')



    # op
    df = df.merge(gen_user_tfidf_features(df=op, value='op_mode'), on=['user'], how='left')
    df = df.merge(gen_user_tfidf_features(df=op, value='op_type'), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_mode'), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_type'), on=['user'], how='left')




    # LabelEncoder
    cat_cols = []
    for col in tqdm([f for f in df.select_dtypes('object').columns if f not in ['user']]):
        le = LabelEncoder()
        df[col].fillna('-1', inplace=True)
        df[col] = le.fit_transform(df[col])
        cat_cols.append(col)

    return df


def lgb_model(train, target, test, k):

    feats = [f for f in train.columns if f not in ['user', 'label']]
    print('Current num of features:', len(feats))
    
    oof_probs = np.zeros(train.shape[0])
    output_preds = 0
    offline_score = []
    feature_importance_df = pd.DataFrame()
    parameters = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 68,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 25,
        'verbose': -1,
        'nthread': 8,
        'max_depth':8
    }

    seeds = [2020]
    for seed in seeds:
        folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for i, (train_index, test_index) in enumerate(folds.split(train, target)):
            train_y, test_y = target[train_index], target[test_index]
            train_X, test_X = train[feats].iloc[train_index, :], train[feats].iloc[test_index, :]

            dtrain = lgb.Dataset(train_X,
                                 label=train_y)
            dval = lgb.Dataset(test_X,
                               label=test_y)
            lgb_model = lgb.train(
                    parameters,
                    dtrain,
                    num_boost_round=5000,
                    valid_sets=[dval],
                    early_stopping_rounds=200,
                    verbose_eval=100,
            )
            oof_probs[test_index] = lgb_model.predict(test_X[feats], num_iteration=lgb_model.best_iteration)/len(seeds)
            offline_score.append(lgb_model.best_score['valid_0']['auc'])
            output_preds += lgb_model.predict(test[feats], num_iteration=lgb_model.best_iteration)/folds.n_splits/len(seeds)
            print(offline_score)
            # feature importance
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = lgb_model.feature_importance(importance_type='gain')
            fold_importance_df["fold"] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('OOF-MEAN-AUC:%.6f, OOF-STD-AUC:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    print('feature importance:')
    print(feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False).head(310))
    feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False).head(457).to_csv('../importance/08_26_452.csv')
    return output_preds, oof_probs, np.mean(offline_score)

if __name__ == '__main__':
    DATA_PATH = '../data/'
    print('读取数据...')
    data, op_df, trans_df = data_preprocess(DATA_PATH=DATA_PATH)

    print('开始特征工程...')
    data = gen_features(data, op_df, trans_df)
    data['city_level'] = data['city'].map(str) + '_' + data['level'].map(str)
    data['city_balance_avg'] = data['city'].map(str) + '_' + data['balance_avg'].map(str)
    data['city_age'] = data['city'].map(str) + '_' + data['age'].map(str)
    data['city_verified'] = data['city'].map(str) + '_' + data['verified'].map(str)
    data['city_agreement_total'] = data['city'].map(str) + '_' + data['agreement_total'].map(str)
    data['city_acc_count'] = data['city'].map(str) + '_' + data['acc_count'].map(str)
    data['province_age'] = data['province'].map(str) + '_' + data['age'].map(str)
    data['province_verified'] = data['province'].map(str) + '_' + data['verified'].map(str)
    data['province_acc_count'] = data['province'].map(str) + '_' + data['acc_count'].map(str)

    print('开始模型训练...')
    train = data[~data['label'].isnull()].copy()
    target = train['label']
    test = data[data['label'].isnull()].copy()

    target_encode_cols = ['province', 'city', 'city_level', 'city_balance_avg','city_age','city_verified','city_agreement_total','agreement_total','city_acc_count','province_age','province_verified','province_acc_count']
    train, test = kfold_stats_feature(train, test, target_encode_cols, 5)
    train.drop(['city_level', 'city_balance_avg','city_age','city_verified','city_agreement_total','city_acc_count','province_age','province_verified','province_acc_count'], axis=1, inplace=True)
    test.drop(['city_level', 'city_balance_avg','city_age','city_verified','city_agreement_total','city_acc_count','province_age','province_verified','province_acc_count'], axis=1, inplace=True)



    lgb_preds, lgb_oof, lgb_score= lgb_model(train=train, target=target, test=test, k=5)
    test['balance_avg_service3'] = test['balance_avg'].map(str) + '_' + test['service3'].map(str)
    a = test[test['balance_avg_service3'] == 'level 21_category 1']['user'].values.tolist()
    for i in range(a):
        lgb_score=round(lgb_score,5)
        sub_df = test[['user']].copy()
        sub_df['prob'] = lgb_preds
        off = test[['user']].copy()
        subVal_df=train[['user']].copy()
        subVal_df['prob']=lgb_oof
        outpath='../submission/'
        sub_df.to_csv(outpath+str(lgb_score)+'_'+time+'sub.csv', index=False)
        subVal_df.to_csv(outpath+str(lgb_score)+'_'+time+'subVal.csv', index=False)