from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


#stop = []
 #   with open("stop-words-russian.txt") as f:
  #      stop = f.read().split("\n")
    # классификация

def load()
    # загрузка обученной модели
    with open('text_classifier', 'rb') as training_model:
        trained_model = pickle.load(training_model)
    with open('vectorizer', 'rb') as vct:
        vectorizer1 = pickle.load(vct)



def file_classifier(X_test):
    target = []
    with open(X_test) as fl:
        for line in fl:
            if line.startswith("CONTENT:"):
                target.append(line[line.index(":") + 1:])

    y_pred = trained_model.predict(vectorizer1.transform([X_test]))

    return y_pred


def text_classifier(X_test):
    # загрузка обученной модели
    with open('text_classifier', 'rb') as training_model:
        trained_model = pickle.load(training_model)
    # загрузка векторайза
    with open('vectorizer', 'rb') as vct:
        vectorizer1 = pickle.load(vct)
    # заполнение словаря

    y_pred = trained_model.predict(vectorizer1.transform([X_test]))

    return y_pred

print(text_classifier('Крупнейшей в России дата-центр «Менделеев», созданный при поддержке концерна «Росэнергоатом» и компании «Ростелеком», открыли в Удомельском районе Тверской области. Презентация этого центра состоялась, сейчас он открыт и поэтапно запускается в промышленную эксплуатацию. И мы надеемся, что центр в ближайшее время выйдет на большой объем хранения данных.На первом этапе в дата-центре введено около 4,8 тыс. серверных стоек, затем их количество увеличат до 12,8 тыс. В пресс-службе правительства области уточнили, что эта работа должна завершиться в 2018 году. «В регионе развивается система „Безопасный город“, ведется информатизация экономики и сферы социальных услуг. Планируется, что в дата-центр будут переведены важнейшие государственные информационные системы», — добавили в пресс-службе.Как сообщалось, строительство дата-центра рядом с Калининской атомной электростанцией (КАЭС) началось в 2016 году. Инвестиционный проект реализуется в рамках программы «Развитие мощностей Центра обработки и хранения данных АО «Концерн Росэнергоатом» в партнерстве с компанией «Ростелеком». Центр будет предоставлять крупным корпорациям комплекс IT-услуг по защите персональных данных, хранению необходимой информации.'))