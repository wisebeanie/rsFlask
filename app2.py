from flask import Flask
from flask import request
import sys
from unittest import result
import warnings
import numpy as np
import pandas as pd
import surprise
import json
from sklearn.decomposition import NMF # Use this for training Non-negative Matrix Factorization
from sklearn.utils.extmath import randomized_svd # Use this for training Singular Value Decomposition
from sklearn.manifold import TSNE # Use this for training t-sne manifolding
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def getData(skins):
    dir = '화장품 추천시스템/최종데이터/'
    df_product = pd.read_csv(dir + 'basic_data_img.csv', usecols=['00.상품코드','00.상품_URL','00.이미지_URL','01.브랜드','02.상품명','03.가격','04.제품 주요 사양','05.모든 성분','06.총 평점','07.리뷰 개수','08_1.별점 1점','08_2.별점 2점','08_3.별점 3점','08_4.별점 4점','08_5.별점 5점','09_1.피부타입_건성','09_2.피부타입_복합성','09_3.피부타입_지성','10_1.피부고민_보습','10_2.피부고민_진정','10_3.피부고민_주름/미백','11_1.피부자극_없음','11_2.피부자극_보통','11_3.피부자극_있음'],encoding='cp949')
    df_review = pd.read_csv(dir + 'total_review.csv', usecols=['code','user','type','tone','problem','rating','feature','review','total_rating'],encoding='cp949')

    user_review_count = df_review['user'].value_counts()
    user_review_count = pd.DataFrame(user_review_count)
    user_review_count = user_review_count.reset_index()
    user_review_count.columns = ['user','count']

    df_review_count = pd.merge(df_review,user_review_count,on='user',how='left')
    df_review_count = df_review_count[df_review_count['count']>=2]
    
    #사용자가 입력한 속성과 가장 비슷한 기존 유저를 뽑는 작업
    df_user_feature = df_review_count[['user','type','tone','problem']]
    df_user_feature = df_user_feature.drop_duplicates(['user'])

    #user의 통합 feature를 담을 데이터 프레임
    df_feature = pd.DataFrame(columns = ['user','feature'])

    for index in df_user_feature.index:
        user = df_user_feature.loc[index]['user']
        problems = df_user_feature.loc[index]['problem']
        problems_list = problems.split('|')
        
        problem = ''
        for i in problems_list:
            i = i.replace(" ","")
            problem += (i+' ')
        
        feature = str(df_user_feature.loc[index]['type']) +' '+str(df_user_feature.loc[index]['tone']) + ' '+problem
        
        df_feature.loc[str(index)] = [user,feature]
        
    #사용자가 입력한 정보를 df_feature 맨 아래에 추가  
    df_feature.loc[str(index)] = ('input',skins)
    df_feature = df_feature.reset_index(drop=True)
    
    counter_vector = CountVectorizer(ngram_range=(1,3))
    c_vector_features = counter_vector.fit_transform(df_feature['feature'])
    
    similarity_feature = cosine_similarity(c_vector_features,c_vector_features).argsort()[:,::-1]

    def recommend_user_list(df_feature, user , top=3):
        #특정 제품코드 뽑아내기
        target_feature_index = df_feature[df_feature['user'] == user].index.values

        #타켓제품과 비슷한 코사인 유사도값
        sim_index = similarity_feature[int(target_feature_index),:top].reshape(-1)
        #본인제외
        sim_index = sim_index[sim_index!=target_feature_index]
        
        #추천 결과 새로운 df 생성, 평균평점(score)으로 정렬
        result = df_feature.iloc[sim_index]
        
        return result
    
    df_result = recommend_user_list(df_feature, user='input')
    similar_user = df_result['user']
    similar_user = similar_user.reset_index(drop=True)
    similar_user = similar_user[0]
    
    A = df_review_count.pivot_table(index = 'code', columns = 'user',values = 'total_rating')
    A = A.copy().fillna(0)  
    
    final_df = df_review_count[['user','code','total_rating']]

    reader = surprise.Reader(rating_scale = (1,5))

    col_list = ['user','code','total_rating']
    data = surprise.Dataset.load_from_df(final_df[col_list], reader)
    trainset = data.build_full_trainset()
    option = {'name' : 'pearson'}
    algo = surprise.KNNBasic(sim_options = option)

    algo.fit(trainset)
    
    name_list = final_df['user'].unique()
    name_list = pd.Series(name_list)

    index = name_list[name_list == similar_user].index[0]
    
    name_list = final_df['user'].unique()
    name_list = pd.Series(name_list)
    
    result = algo.get_neighbors(index,k=5)
    
    code_list = []
    for r1 in result:
        max_rating = data.df[data.df['user']==name_list[r1]]['total_rating'].max()
        cos_id = data.df[(data.df['total_rating']==max_rating)&(data.df['user']==name_list[r1])]['code'].values
        
        code_list.append(cos_id)
    
    result_dict={}
    products_dict = {}
    i = 0
    for codes in code_list:
        for code in codes:
    #         print(df_product[df_product['00.상품코드']==code]['00.상품_URL'].item())
    #         print(df_product[df_product['00.상품코드']==code]['02.상품명'].item())
            
            product_dict = {}
            product_dict['productURL'] = str(df_product[df_product['00.상품코드']==code]['00.상품_URL'].item())
            product_dict['imageURL'] = str(df_product[df_product['00.상품코드']==code]['00.이미지_URL'].item())
            product_dict['brand'] = str(df_product[df_product['00.상품코드']==code]['01.브랜드'].item())
            product_dict['productName'] = str(df_product[df_product['00.상품코드']==code]['02.상품명'].item())
            product_dict['price'] = int(df_product[df_product['00.상품코드']==code]['03.가격'].item())
            
            products_dict[i] = product_dict
            i+=1
            if i==10:
                break
        if i==10:
            break
        
    result_dict['CF'] = products_dict
    
    return result_dict

@app.route('/cf')
def cf():
    
    skin_type = request.args.get('skintype')
    skin_tone = request.args.get('skintone')
    skin_worries = request.args.getlist('skinworry')

    skins = skin_type + ' ' + skin_tone
    
    for skin_worries in skin_worries:
        skins = skins + ' ' + skin_worries
        
    print(skins)

    return getData(skins)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000, debug=True)