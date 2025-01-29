from flask import Flask, request, jsonify, render_template_string, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import datetime
import datetime as dt

app = Flask(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("tasarim.html") # Doğru kullanım # Form sayfasını göster

@app.after_request
def after_request(response):
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response


@app.route('/process_form', methods=['POST'])
def process_form():
    # Form verilerini al
    marka = request.form.get('marka')
    print(marka)
    model = request.form.get('model')
    print(model)
    yil = int(request.form.get('year'))
    print(yil)
    km = int(request.form.get('km'))
    print(km)
    girdi = request.form.get('vites_turu')  # String olarak gelir
    #print(vites_turu)
    #print(girdi)
    if girdi == "1":
        vites_turu = "Duz"
    elif girdi == "2":
        vites_turu = "Otomatik"
    elif girdi == "3":
        vites_turu = "Yari Otomatik"
    else:
        print("Vites türü hatalı geldi:", girdi)

    girdi2 = request.form.get('fuel')  # String olarak gelir
    ##print(yakit_turu)
    #print(girdi2)
    if girdi2 == "1":
        yakit_turu = "Benzin"
    elif girdi2 == "2":
        yakit_turu = "Dizel"
    elif girdi2 == "3":
        yakit_turu = "LPG & Benzin"
    else:
        print("Yakit türü hatalı geldi:", girdi2)

    renk = request.form.get('renk')

    print(vites_turu)
    print(yakit_turu)
    print(renk)
    # Kullanıcıdan gelen veriyi bir DataFrame olarak düzenle
    dataSayfaGelen = {
        'Yıl': [yil],
        'Kilometre': [km],
        'Vites Tipi': [vites_turu],
        'Yakıt Tipi': [yakit_turu],
        'Renk': [renk]
    }
    dfSayfaGelen = pd.DataFrame(dataSayfaGelen)

    # Eğitim verisi (Web Kazıma ile)
    BASE_URL = "https://www.arabam.com"
    URL = f"{BASE_URL}/ikinci-el?searchText={model}&take=50"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }

    sayfa = requests.get(URL, headers=headers)
    icerik = BeautifulSoup(sayfa.content, "html.parser")
    pagination_class_ici = icerik.find('ul', class_='pagination')
    soup = BeautifulSoup(sayfa.content, "html.parser", from_encoding="utf-8")
    son_sayfa = int(pagination_class_ici.find_all("a")[-2].text.strip()) if pagination_class_ici else 1

    # Sayfa bağlantılarını oluştur
    sayfaBaglantilari = [f"{BASE_URL}/ikinci-el?searchText={model}&take=50&page={i}" for i in range(1, son_sayfa + 1)]

    # İlan bağlantılarını getir
    def arabaLinkleriniGetir(url):
        sayfa = requests.get(url, headers=headers)
        soup = BeautifulSoup(sayfa.text, 'html.parser')
        arabaLinkleri = {BASE_URL + link.get('href') for link in soup.find_all('a', class_='link-overlay')}
        return arabaLinkleri

    # İlan özelliklerini getir
    def arabaOzellikleri(url):
        try:
            sayfa = requests.get(url, headers=headers)
            icerik = BeautifulSoup(sayfa.content, "html.parser")
            fiyat = icerik.find('div', {'data-testid': 'desktop-information-price'})
            fiyat = int(fiyat.get_text(strip=True).replace("TL", "").replace(".", "").strip()) if fiyat else None
            ozellikler = icerik.find_all('div', class_='property-item')
            araba_bilgileri = {'Fiyat': fiyat}
            gerekli_ozellikler = ["Yıl", "Kilometre", "Vites Tipi", "Yakıt Tipi", "Renk"]
            for ozellik in ozellikler:
                key = ozellik.find('div', class_='property-key').get_text(strip=True)
                value = ozellik.find('div', class_='property-value').get_text(strip=True)
                if key in gerekli_ozellikler:
                    araba_bilgileri[key] = value
            return araba_bilgileri
        except Exception as e:
            print(f"Hata oluştu: {e} (URL: {url})")
            return {}

    # İlanları topla
    ilanLinkleri = []
    for sayfaLinki in tqdm(sayfaBaglantilari, desc="Sayfalar İşleniyor"):
        ilanLinkleri.extend(arabaLinkleriniGetir(sayfaLinki))

    with ThreadPoolExecutor() as executor:
        ilanDetaylari = list(tqdm(executor.map(arabaOzellikleri, ilanLinkleri), total=len(ilanLinkleri), desc="İlanlar İşleniyor"))



    # Veriyi temizle ve düzenle
    df = pd.DataFrame(ilanDetaylari).dropna()
    print(df.head(5))



    df["Kilometre"] = df["Kilometre"].str.replace(" km", "").str.replace(".", "").astype(int)
    df["Vites Tipi"] = df["Vites Tipi"].str.replace("ü", "u") #Düz -> Duz
    df["Vites Tipi"] = df["Vites Tipi"].str.replace("ı", "i") #Yarı otomatik -> Yari otomatik
    print(df.head(5))

    # Pandas DataFrame'e dönüştür ve CSV'ye kaydet
    dosyaAdi = model
    df.to_csv(f"{dosyaAdi}.csv", index=False, encoding='utf-8-sig')
    print(f"Veriler {dosyaAdi} dosyasına başarıyla kaydedildi.")

    # Kategorik verileri encode et
    # Kategorik verileri encode et
    label_encoders = {}
    for column in ['Vites Tipi', 'Yakıt Tipi', 'Renk']:
        le = LabelEncoder()

        # Eğitim verisini encode et
        df[column] = le.fit_transform(df[column])

        # LabelEncoder nesnesini sakla
        label_encoders[column] = le

        # Yeni veriyi encode et (bilinmeyen değerler için en sık görüleni kullan)
        if dfSayfaGelen[column][0] in le.classes_:
            dfSayfaGelen[column] = le.transform(dfSayfaGelen[column])
        else:
            most_frequent = pd.Series(le.classes_).mode()[0]
            print(
                f"Bilinmeyen değer: {dfSayfaGelen[column][0]} ({column} sütunu için). En sık görünen değere ({most_frequent}) atanıyor.")
            dfSayfaGelen[column] = le.transform([most_frequent] * len(dfSayfaGelen[column]))

    df['Yıl'] = df['Yıl'].astype(int)  # Bu satır eklendi!
    df['Araba_Yasi'] = datetime.datetime.now().year - df['Yıl']
    dfSayfaGelen['Araba_Yasi'] = datetime.datetime.now().year - dfSayfaGelen['Yıl']
    df = df.drop('Yıl', axis=1)  # Yıl kolonunu düşür
    dfSayfaGelen = dfSayfaGelen.drop('Yıl', axis=1)  # Yıl kolonunu düşür

    # Korelasyonu hesaplayın
    correlation_matrix = df.corr()

    # Fiyat sütunu ile diğer sütunlar arasındaki korelasyon
    correlation_with_price = correlation_matrix['Fiyat']

    # Korelasyonu yazdırın
    print(correlation_with_price)

    # Min-Max ölçekleyici oluştur
    scaler_km = MinMaxScaler()
    df['Kilometre'] = scaler_km.fit_transform(df[['Kilometre']])
    dfSayfaGelen['Kilometre'] = scaler_km.transform(dfSayfaGelen[['Kilometre']])  # Eğitilen scaler'ı kullanıyoruz.#

    # Veriyi train-test split yap
    x = df.drop("Fiyat", axis=1)
    y = df["Fiyat"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

    # Modeli eğit
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train)

    # Model performansını ölç
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(df.head(5))
    print(f"Model Performans: MSE = {mse:.2f}, R2 = {r2:.2f}")


    # Tahmin yap 
    tahmin = model.predict(dfSayfaGelen)
    print(tahmin)

    # Tahmin sonucunu düzelt ve yuvarla
    tahminSade = round(tahmin[0], 2)

    return jsonify({'tahmin': tahminSade, 'r2': r2})


if __name__ == '__main__':
    app.run(debug=True)
