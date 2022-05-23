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

def getCf(skins):
    dir = '화장품 추천시스템/최종데이터/'
    df_product = pd.read_csv(dir + 'basic_data_img.csv', usecols=['00.상품코드','00.상품_URL','00.이미지_URL','01.브랜드','02.상품명','03.가격','04.제품 주요 사양','05.모든 성분','06.총 평점','07.리뷰 개수','08_1.별점 1점','08_2.별점 2점','08_3.별점 3점','08_4.별점 4점','08_5.별점 5점','09_1.피부타입_건성','09_2.피부타입_복합성','09_3.피부타입_지성','10_1.피부고민_보습','10_2.피부고민_진정','10_3.피부고민_주름/미백','11_1.피부자극_없음','11_2.피부자극_보통','11_3.피부자극_있음'],encoding='cp949')
    df_review = pd.read_csv(dir + 'total_review.csv', usecols=['code','user','type','tone','problem','rating','feature','review','total_rating'],encoding='cp949')

    user_review_count = df_review['user'].value_counts()
    user_review_count = pd.DataFrame(user_review_count)
    user_review_count = user_review_count.reset_index()
    user_review_count.columns = ['user','count']

    df_review_count = pd.merge(df_review,user_review_count,on='user',how='left')
    df_review_count = df_review_count[df_review_count['count']>=2]
    df_review_count
    
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
    df_feature
    df_feature.loc[str(index)] = ('input',skins)
    df_feature = df_feature.reset_index(drop=True)
    print(df_feature)
    
    counter_vector = CountVectorizer(ngram_range=(1,3))
    c_vector_features = counter_vector.fit_transform(df_feature['feature'])
    c_vector_features.shape
    
    similarity_feature = cosine_similarity(c_vector_features,c_vector_features).argsort()[:,::-1]
    print(similarity_feature)
    print(similarity_feature.shape)
    
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
    similar_user
    
    A = df_review_count.pivot_table(index = 'code', columns = 'user',values = 'total_rating')
    A = A.copy().fillna(0)
    
    final_df = df_review_count[['user','code','total_rating']]
    final_df
    
    reader = surprise.Reader(rating_scale = (1,5))

    col_list = ['user','code','total_rating']
    data = surprise.Dataset.load_from_df(final_df[col_list], reader)

    #surprise KNN 기반 CF 진행

    #학습
    trainset = data.build_full_trainset()
    option = {'name' : 'pearson'}
    algo = surprise.KNNBasic(sim_options = option)

    algo.fit(trainset)
    
    name_list = final_df['user'].unique()
    name_list = pd.Series(name_list)

    index = name_list[name_list == similar_user].index[0]
    index
    
    result = algo.get_neighbors(index,k=5)
    print('해당 사용자와 가장 유사한 사용자들은?', result)
    print(name_list[result])
    
    code_list = []
    for r1 in result:
        max_rating = data.df[data.df['user']==name_list[r1]]['total_rating'].max()
        cos_id = data.df[(data.df['total_rating']==max_rating)&(data.df['user']==name_list[r1])]['code'].values
        
        code_list.append(cos_id)
    code_list
    
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

def getCbf(skinList):
    dir = '화장품 추천시스템/최종데이터/'
    df_product = pd.read_csv(dir + 'basic_data_img.csv', usecols=['00.상품코드','00.상품_URL','00.이미지_URL','01.브랜드','02.상품명','03.가격','04.제품 주요 사양','05.모든 성분','06.총 평점','07.리뷰 개수','08_1.별점 1점','08_2.별점 2점','08_3.별점 3점','08_4.별점 4점','08_5.별점 5점','09_1.피부타입_건성','09_2.피부타입_복합성','09_3.피부타입_지성','10_1.피부고민_보습','10_2.피부고민_진정','10_3.피부고민_주름/미백','11_1.피부자극_없음','11_2.피부자극_보통','11_3.피부자극_있음'],encoding='cp949')
    df_review = pd.read_csv(dir + 'total_review.csv', usecols=['code','user','type','tone','problem','rating','feature','review','total_rating'],encoding='cp949')
    
    product_list = df_review['00.상품코드']
    product_list = product_list.unique()
    len(product_list)
    
    df_review.columns = ['code','user','type','tone','problem','rating','feature','review']
    df_review = df_review.dropna(axis=0)
    
    df_product.columns = ['code','00.상품_URL','00.이미지_URL','01.브랜드','02.상품명','03.가격','04.제품 주요 사양','05.모든 성분','06.총 평점','07.리뷰 개수','08_1.별점 1점','08_2.별점 2점','08_3.별점 3점','08_4.별점 4점','08_5.별점 5점','09_1.피부타입_건성','09_2.피부타입_복합성','09_3.피부타입_지성','10_1.피부고민_보습','10_2.피부고민_진정','10_3.피부고민_주름/미백','11_1.피부자극_없음','11_2.피부자극_보통','11_3.피부자극_있음']
    df_product = df_product.dropna(axis=0)
    df_product = df_product.drop_duplicates(['code'])
    
    user_list = df_review['code']
    user_list = user_list.unique()
    user_list[0]
    df_review_user = df_review[df_review['code']==user_list[0]]

    problem_dict = {}
    for problem in df_review_user['problem']:
        problem_list = problem.split('|')
        for i in problem_list:
            i = i.replace(" ","")
            if i in problem_dict:
                problem_dict[i] += 1
            else:
                problem_dict[i] = 1
    problem_dict_sort = sorted(problem_dict.items(),key = lambda x: x[1],reverse = True)
    problem_dict_sort[0][0]
    
    user_list = df_review['code']
    user_list = user_list.unique()
    user_list[0]
    df_review_user = df_review[df_review['code']==user_list[0]]

    #최다빈출 type
    type_freq = df_review_user['type'].value_counts()
    #최다빈출 tone
    tone_freq = df_review_user['tone'].value_counts()
    #최다빈출 problem
    problem_dict = {}
    for problem in df_review_user['problem']:
        problem_list = problem.split('|')
        for i in problem_list:
            i = i.replace(" ","")
            if i in problem_dict:
                problem_dict[i] += 1
            else:
                problem_dict[i] = 1
    problem_dict_sort = sorted(problem_dict.items(),key = lambda x: x[1],reverse = True)



    type = type_freq.index[0]
    tone = tone_freq.index[0]
    problem = ''
    for i, problems in enumerate(problem_dict_sort):
        problem += problems[0] + ' '
        if i==20:
            break
    df_feature = pd.DataFrame(columns = ['code','feature'])


    product_list = df_review['code']
    product_list = product_list.unique()
    index = 0
    for product in product_list:
        df_review_user = df_review[df_review['code']==product]
        
        #최다빈출 type
        type_freq = df_review_user['type'].value_counts()
        #최다빈출 tone
        tone_freq = df_review_user['tone'].value_counts()
        #최다빈출 problem
        problem_dict = {}
        for problem in df_review_user['problem']:
            problem_list = problem.split('|')
            for i in problem_list:
                i = i.replace(" ","")
                if i in problem_dict:
                    problem_dict[i] += 1
                else:
                    problem_dict[i] = 1
        problem_dict_sort = sorted(problem_dict.items(),key = lambda x: x[1],reverse = True)



        type = type_freq.index[0]
        tone = tone_freq.index[0]
        problem = ''
        for i, problems in enumerate(problem_dict_sort):
            problem += problems[0] + ' '
            if i==4:
                break

        feature = type+' '+tone+' '+problem
        
        df_feature.loc[str(index)] = [product,feature]
    
        index += 1
        
    df_feature.loc[str(index)] = ('input','건성 웜톤 트러블 미백 주름')
    
    df_feature.to_csv(dir+'df_feature.csv',header = True, index = True,encoding='cp949')
    
    counter_vector = CountVectorizer(ngram_range=(1,3))
    c_vector_features = counter_vector.fit_transform(df_feature['feature'])
    c_vector_features.shape
    
    similarity_feature = cosine_similarity(c_vector_features,c_vector_features).argsort()[:,::-1]
    print(similarity_feature)
    print(similarity_feature.shape)
    
    def recommend_product_list(df_feature, code , top=11):
        #특정 제품코드 뽑아내기
        target_feature_index = df_feature[df_feature['code'] == code].index.values

        #타켓제품과 비슷한 코사인 유사도값
        sim_index = similarity_feature[int(target_feature_index),:top].reshape(-1)
        #본인제외
        sim_index = sim_index[sim_index!=target_feature_index]
        
        #추천 결과 새로운 df 생성, 평균평점(score)으로 정렬
        result = df_feature.iloc[sim_index]
        
        return result
    
    df_result = recommend_product_list(df_feature, code='input')
    
    df_recommend = pd.merge(df_product,df_result,on='code',how='right')
    df_recommend

    result_dict={}
    products_dict = {}

    for index in df_recommend.index:
        if index == 0:
            continue
            
        product_dict = {}
        product_dict['productURL'] = str(df_recommend.loc[index]['00.상품_URL'])
        product_dict['imageURL'] = str(df_recommend.loc[index]['00.이미지_URL'])
        product_dict['brand'] = str(df_recommend.loc[index]['01.브랜드'])
        product_dict['productName'] = str(df_recommend.loc[index]['02.상품명'])
        product_dict['price'] = int(df_recommend.loc[index]['03.가격'])

        products_dict[index] = product_dict
        
    result_dict['CBF'] = products_dict

    return result_dict

@app.route('/cf')
def cf():
    skin_type = request.args.get('skintype')
    skin_tone = request.args.get('skintone')
    skin_worries = request.args.getlist('skinworry')

    skins = skin_type + ' ' + skin_tone
    
    for skin_worries in skin_worries:
        skins = skins + ' ' + skin_worries
        
    return getCf(skins)

@app.route('/cbf')
def cbf():
    # skin_type = request.args.get('skintype')
    # skin_tone = request.args.get('skintone')
    # skin_worries = request.args.getlist('skinworry')
    
    # skins = skin_type + ' ' + skin_tone
    skins = ''
    # for skin_worries in skin_worries:
    #     skins = skins + ' ' + skin_worries
        
    return getCbf(skins)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000, debug=True)