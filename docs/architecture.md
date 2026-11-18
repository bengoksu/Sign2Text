# Sistem Mimarisi

## Kapsam ve temel sınır

Sign2Text'in ilk sürümü iki ayrı yapay zekâ problemini ardışık çözecektir:

1. Tek el görüntüsünden ASL parmak alfabesi sınıfını tahmin etme.
2. Oluşan ASCII tabanlı ham metinde Türkçe karakterleri ve sınırlı yazım
   bozukluklarını bağlama göre restore etme.

Bu sistem TİD tanıma, sürekli işaret dili tanıma veya işaret dili cümle çevirisi
olarak sunulmayacaktır.

## Bileşenler ve veri sözleşmeleri

```text
CameraFrame (BGR)
    |
    v
HandTracker ----> DetectionStatus
    |
    v
HandCrop (RGB) -> ImagePreprocessor -> ModelTensor
    |
    v
ASLRecognizer -> Prediction(label, probability, timestamp)
    |
    v
PredictionStabilizer -> StablePrediction
    |
    v
ExplicitConfirmation / guarded control gesture
    |
    v
TextComposer -> RawText
    |
    v  (yalnız kelime/cümle tamamlanınca veya kullanıcı isteğiyle)
TurkishTextCorrector -> CorrectionResult(raw, corrected, suggestions, errors)
    |
    v
Streamlit UI
```

Modüller kamera karesini, tahmin sonucunu ve metni küresel değişkenlerle değil,
açık veri nesneleri üzerinden taşıyacaktır. Böylece her bileşen bağımsız test
edilebilir.

## Dizin sorumlulukları

| Dizin | Sorumluluk | Planlanan aşama |
|---|---|---:|
| `config/` | Yollar, sınıflar ve uygulama ayarları | 1 |
| `recognition/` | Veri yükleme, ön işleme, eğitim, değerlendirme, tahmin, kamera ve el tespiti | 2-4 |
| `text_builder/` | Smoothing, debounce, açık onay ve metin işlemleri | 5 |
| `nlp/` | Model yükleme, normalizasyon, restorasyon ve öneriler | 6 |
| `app.py` | Streamlit kompozisyon kökü | 7 |
| `tests/` | Birim, entegrasyon ve uçtan uca testler | tüm aşamalar |
| `docs/` | Deney tasarımı, veri ve sonuç kayıtları | tüm aşamalar |

## Görüntü tanıma kararı

Ana veri seti piksel görüntülerinden oluştuğu için ilk deney hattı görüntü tabanlı
olacaktır. MediaPipe yalnızca el bölgesini bulmak için kullanılabilir; landmark
koordinatları, piksel CNN'ine doğrudan özellik olarak verilmez. Landmark tabanlı
bir alternatif denenirse aynı landmark özelliklerinden oluşturulmuş ayrı bir veri
seti ve ayrı bir sınıflandırıcı gerekir.

Aşama 3'te küçük bir özel CNN ile CPU dostu transfer öğrenme modeli aynı veri
bölmeleri üzerinde karşılaştırılacaktır. Seçim yalnız doğrulukla değil makro F1,
model boyutu, CPU gecikmesi ve gerçek kamera deneyiyle yapılacaktır.

Eğitim ve canlı tahmin aynı renk uzayı, kırpma/padding, yeniden boyutlandırma ve
ölçekleme sözleşmesini paylaşacaktır. Bu dönüşümler tek bir `preprocessing.py`
uygulamasında tutulacaktır.

## Veri bölme ve sızıntı kontrolü

Yayıncının test klasöründe gerçekte 28 görüntü vardır ve tamamı eğitim klasöründeki
bir dosyanın kesin kopyasıdır; bağımsız değerlendirmede kullanılmayacaktır. Kişi
ve oturum kimliği bulunmadığı için dosya sırasından türetilen 50 karelik gruplar
%70/%15/%15 bölünmüş, aynı grubun bölümler arasında geçmesi engellenmiştir. Bu
sezgisel önlem kişi bağımsızlığı sağlamaz. Nihai değerlendirmede proje ekibinin
farklı ışık ve arka planlarda çektiği, eğitimden tamamen ayrı kamera verisi
bulunacaktır.

`J` ve `Z` hareket gerektirir. İlk statik modelin bu sınıflardaki çıktısı yalnız
tek kare poz benzerliği anlamına gelir. Gerçek hareket tanıma için zaman pencereli
landmark/görüntü dizisi ve sekans modeli ayrı bir geliştirme olarak ele alınır.

## Gerçek zamanlı metin oluşturma

Her kare metne eklenmeyecektir. Tahminler güven eşiği ve kayan pencere ile
kararlı hale gelecek, bir sınıf yeterli süre/oy sayısına ulaştığında aday olacaktır.
İlk sürümde kullanıcı açıkça onaylayacaktır. Aynı harfi tekrar yazmak için önce
nötr duruma (`nothing` veya el yok) dönme ya da yeni bir açık onay döngüsü gerekir.

`space` ve `del` normal harften daha sıkı eşik ve bekleme süresi kullanacak;
tek karelik tahmin işlem çalıştırmayacaktır. Arayüz düğmeleri her zaman güvenli
alternatif olarak bulunacaktır.

## NLP sınırı

`TurkishTextCorrector` kamera ve Streamlit'ten bağımsız olacaktır. Ham metni asla
üzerine yazmayacak; yapılandırılmış sonuç döndürecektir. Model yalnız kullanıcı
isteğinde veya kelime/cümle sınırında çağrılacaktır. Model seçimi Aşama 6'ya
bırakılmıştır; model varlığı, lisansı, CPU belleği/gecikmesi ve ayrılmış örneklerde
CER/WER ölçülmeden seçim yapılmayacaktır.

## Hata ve kaynak yönetimi

- Modeller ihtiyaç anında bir kez yüklenecek ve açık yükleme durumu sunacaktır.
- Kamera kaynağı başlatma/durdurma sınırlarında serbest bırakılacaktır.
- Model/veri dosyası eksikliği kullanıcıya çözüm komutuyla bildirilecektir.
- FPS ve gecikme yalnız gerçek ölçümden üretilecek; sabit değer gösterilmeyecektir.
- Büyük veri ve model dosyaları Git dışında tutulacaktır.

