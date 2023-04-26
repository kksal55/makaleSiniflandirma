
# Gerekli kütüphaneleri ve modülleri içeri aktarma
import jpype as jp, nltk, pandas as pd, numpy as np, veriSeti
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# nltk'nin stopwords paketini indirme
nltk.download('stopwords')

# Zemberek kütüphanesinin yolu
zemberekYol = 'zemberek-full.jar'

# Varsayılan girdi metni
girdiMetinVarsayilan = """	
Sağlık, insan yaşamının en temel ve değerli unsurlarından biridir. Sağlıklı olmak, kişinin fiziksel, zihinsel ve sosyal açıdan tam bir iyilik halinde olması demektir. Sağlık, bireylerin hayat kalitesini artıran ve üretkenliklerini artıran önemli bir faktördür.Sağlığı etkileyen birçok faktör vardır. Beslenme, egzersiz, uyku, stres yönetimi, hijyen gibi faktörler, sağlığı doğrudan etkileyen ana unsurlardan bazılarıdır. Bununla birlikte, çevresel faktörler, genetik faktörler ve yaşam tarzı tercihleri de sağlık üzerinde önemli bir etkiye sahiptir.Sağlıkla ilgili birçok konu vardır. Bunlar arasında hastalıkların teşhisi, tedavisi ve önlenmesi, sağlıklı yaşam tarzı önerileri, beslenme ve egzersiz önerileri, psikolojik ve sosyal sağlık konuları, çocuk sağlığı, kadın sağlığı, erkek sağlığı, yaşlılık sağlığı ve daha birçok konu yer alır.Sağlık konusunda yapılan araştırmaların sayısı da oldukça fazladır. Birçok ülkede sağlık araştırmaları yürüten kuruluşlar, hastaneler, klinikler ve araştırma merkezleri bulunmaktadır. Bu araştırmalar, hastalıkların nedenleri, teşhis yöntemleri, tedavi yöntemleri ve önleme stratejileri gibi konularda bilgi sağlamaktadır. Sağlık, insanlar için çok önemli bir konu olduğundan, sağlık hizmetleri de oldukça önemlidir. Hastaneler, klinikler, doktorlar, hemşireler, diyetisyenler, psikologlar ve daha birçok sağlık uzmanı, insanların sağlık ihtiyaçlarını karşılamak için çalışmaktadır.Sonuç olarak, sağlık konusu oldukça geniş kapsamlı ve önemli bir konudur. Sağlıklı bir yaşam için doğru beslenme, düzenli egzersiz, yeterli uyku ve stres yönetimi gibi faktörlere dikkat etmek gerekmektedir. Ayrıca, sağlık hizmetlerinden yararlanarak, sağlık sorunlarına erken müdahale ederek, sağlıklı bir yaşam sürdürmek mümkündür.
"""

# JVM'yi başlatma ve Zemberek sınıflarını yükleme
jp.startJVM(jp.getDefaultJVMPath(), 'ea', '-Djava.class.path=%s' % (zemberekYol),ignoreUnrecognized=True)
TC_CumleAyirici, TC_Morfoloji, TC_Duzeltici, TC_Tokenlestirici, TC_Lex = (jp.JClass(x) for x in ['zemberek.tokenization.TurkishSentenceExtractor', 'zemberek.morphology.TurkishMorphology', 'zemberek.normalization.TurkishSpellChecker', 'zemberek.tokenization.TurkishTokenizer', 'zemberek.tokenization.antlr.TurkishLexer'])



# Zemberek nesnelerini oluşturma
cumle_ayirici, morfoloji, tokenizer, yazim_kontrol = TC_CumleAyirici.DEFAULT, TC_Morfoloji.createWithDefaults(), TC_Tokenlestirici.ALL, TC_Duzeltici(TC_Morfoloji.createWithDefaults())

# Değişkenlerin tanımlanması
girdiler, ciktilar, tip = [], [], 0



# Token analiz fonksiyonu: Token'in işleme alınıp alınmayacağını belirler
def token_analiz_et(token) -> bool:
    t = token.getType()
    return (t != TC_Lex.NewLine and
            t != TC_Lex.SpaceTab and
            t != TC_Lex.Punctuation and
            t != TC_Lex.RomanNumeral and
            t != TC_Lex.UnknownWord and
            t != TC_Lex.Unknown)

# Metin ön işleme fonksiyonu: Metinde yazım hatalarını düzelten bir fonksiyon
def metin_on_isleme(text):
    tokens = tokenizer.tokenize(text)
    duzeltilmis_metin = ''
    for token in tokens:
        text = token.getText()
        # Eğer token işleme alınması gerekiyorsa ve yazım hatası varsa düzelt
        if (token_analiz_et(token) and not yazim_kontrol.check(text)):
            oneriler = yazim_kontrol.suggestForWord(token.getText())
            if oneriler:
                correction = oneriler.get(0)
                duzeltilmis_metin += str(correction)
            else:
                duzeltilmis_metin += str(text)
        else:
            duzeltilmis_metin += str(text)
    return duzeltilmis_metin

# Kök kelimeleri çıkarma fonksiyonu: Düzeltimiş metindeki kelimelerin köklerini çıkarır
def kok_kelimeleri_cikar(duzeltilmis_metin):
    cumleler = cumle_ayirici.fromParagraph(duzeltilmis_metin)
    kok_kelimeler = []
    for cumle in cumleler:
        analiz = morfoloji.analyzeAndDisambiguate(cumle).bestAnalysis()
        for word in analiz:
            kok_kelimeler.append(word.getLemmas()[0])
    return kok_kelimeler

# Etiketlerin listesi
labels2 = ["bilim","teknoloji", "saglik", "tarih", "ekonomi", "spor"]

# Girdilerin ve çıktıların hazırlanması
girdiler = []
ciktilar = []
engelli_kelimeler = nltk.corpus.stopwords.words('turkish')

for i, category in enumerate(veriSeti.label_data):
    for j in range(0, 21):
        text = category[j]
        duzeltilmis_metin = metin_on_isleme(text)
        kok_kelimeler = kok_kelimeleri_cikar(duzeltilmis_metin)
        kok_kelimeler = [e for e in kok_kelimeler if e not in (',', '.', '"', ";", ":", "?", "!", "$", "#", "/", "UNK", "(", ")")]
        kok_kelimeler = [token for token in kok_kelimeler if token not in engelli_kelimeler]
        kok_kelimeler = str(kok_kelimeler)
        girdiler += [kok_kelimeler]
        ciktilar += [labels2[i]]


# Etiket kodlama nesnesi oluşturma
Encoder = LabelEncoder()
ciktilar = Encoder.fit_transform(ciktilar)

# Veri kümesini eğitim ve test kümesi olarak ayırma
x_deneme, x_test, y_deneme, y_test = model_selection.train_test_split(girdiler, ciktilar, test_size=0.3, random_state=70)

# Tfidf vektörleştirici nesnesi oluşturma
tfidf_vector = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), max_features=10000)
tfidf_vector.fit(girdiler)
Train_X_Tfidf = tfidf_vector.transform(x_deneme)
Test_X_Tfidf = tfidf_vector.transform(x_test)

# Kullanıcının test etmek istediği makaleyi girmesini sağlama
girdiMetin = input("Lütfen test etmek istediğiniz makaleyi girin:(Eğer boş enter'a basarsanız varsayılan giriş metni işlenir)\n\n")
if girdiMetin.strip() == "":
        girdiMetin = girdiMetinVarsayilan
        
# Token analizi ve yazım kontrolü yapma
tokens = tokenizer.tokenize(girdiMetin)
def token_analiz_et (token) -> bool:
    t = token.getType()
    return (t != TC_Lex.NewLine and
            t != TC_Lex.SpaceTab and
            t != TC_Lex.Punctuation and
            t != TC_Lex.RomanNumeral and
            t != TC_Lex.UnknownWord and
            t != TC_Lex.Unknown)


# Token işleme alınması gerekiyorsa ve yazım hatası varsa düzelt
duzeltilmis_metin = ''

for token in tokens:
    text = token.getText()
    if token_analiz_et(token) and not yazim_kontrol.check(text):
        oneriler = yazim_kontrol.suggestForWord(text)
        duzeltilmis_metin += str(oneriler.get(0)) if oneriler else str(text)
    else:
        duzeltilmis_metin += str(text)
        
# Düzgün yazılmış metnin cümlelerini ayırma ve kök kelimeleri çıkarma
cumleler = cumle_ayirici.fromParagraph(duzeltilmis_metin)
kok_kelimeler = []
for cumle in cumleler:
    analiz = morfoloji.analyzeAndDisambiguate(cumle).bestAnalysis()
    for word in analiz:
        kok_kelimeler.append(word.getLemmas()[0])
        
# İşlenmemesi gereken karakterleri ve Türkçe'deki durdurma kelimelerini kaldırma
engelli_kelimeler = nltk.corpus.stopwords.words('turkish')
kok_kelimeler = [e for e in kok_kelimeler if e not in (',', '.', '"', ";", ":", "?", "!", "$", "#", "/", "UNK")]
kok_kelimeler = [token for token in kok_kelimeler if token not in engelli_kelimeler]
kok_kelimeler = str(kok_kelimeler)
girdi = [kok_kelimeler]

# Kullanıcının girdiği metni Tfidf vektörleştirici ile dönüştürme
test = tfidf_vector.transform(girdi)

# Makale türünü döndürme fonksiyonu
def makaleTuruGetir(par):
    types = ["BİLİM&TEKNOLOJİ","BİLİM&TEKNOLOJİ", "SAĞLIK","TARİH", "EKONOMİ","SPOR"]
    return f"{types[par]} MAKALESİ" if 0 <= par < len(types) else "Makale bulunamadı"

# Naive Bayes Algoritması kullanarak modeli eğitme ve tahmin yapma
Naive = MultinomialNB()
Naive.fit(Train_X_Tfidf, y_deneme)
skor = Naive.predict(Test_X_Tfidf)
sonuc = Naive.predict(test)

# Sonuçları ve doğruluk skorunu yazdırma
print("\n")
print("\n")
print("\n")
print("SONUÇ :", makaleTuruGetir(sonuc[0]))
print("DOĞRULUK SKORU: ", accuracy_score(skor, y_test)*100)
print("\n")
jp.shutdownJVM()





girdiMetin = input("Lütfen test etmek istediğiniz makaleyi girin:(Eğer boş ise varsayılan metin işlenir)\n")
if girdiMetin.strip() == "":
        girdiMetin = girdiMetinVarsayilan
        
tokens = tokenizer.tokenize(girdiMetin)
def token_analiz_et (token) -> bool:
    t = token.getType()
    return (t != TC_Lex.NewLine and
            t != TC_Lex.SpaceTab and
            t != TC_Lex.Punctuation and
            t != TC_Lex.RomanNumeral and
            t != TC_Lex.UnknownWord and
            t != TC_Lex.Unknown)

duzeltilmis_metin = ''

for token in tokens:
    text = token.getText()
    if token_analiz_et(token) and not yazim_kontrol.check(text):
        oneriler = yazim_kontrol.suggestForWord(text)
        duzeltilmis_metin += str(oneriler.get(0)) if oneriler else str(text)
    else:
        duzeltilmis_metin += str(text)
        
cumleler = cumle_ayirici.fromParagraph(duzeltilmis_metin)
kok_kelimeler = []
for cumle in cumleler:
    analiz = morfoloji.analyzeAndDisambiguate(cumle).bestAnalysis()
    for word in analiz:
        kok_kelimeler.append(word.getLemmas()[0])
        
engelli_kelimeler = nltk.corpus.stopwords.words('turkish')
kok_kelimeler = [e for e in kok_kelimeler if e not in (',', '.', '"', ";", ":", "?", "!", "$", "#", "/", "UNK")]
kok_kelimeler = [token for token in kok_kelimeler if token not in engelli_kelimeler]
kok_kelimeler = str(kok_kelimeler)
girdi = [kok_kelimeler]
test = tfidf_vector.transform(girdi)