# Sign2Text

Sign2Text, kamerada gösterilen Amerikan İşaret Dili (ASL) parmak alfabesi
işaretlerini harflere dönüştürmeyi ve oluşan ASCII metni ayrı bir NLP bileşeniyle
Türkçe karakterler açısından düzeltmeyi hedefleyen bir üniversite Design Project
çalışmasıdır.

> Proje durumu: **Aşama 2 — veri indirme, tam dosya analizi ve ön işleme
> tamamlandı.** Henüz eğitilmiş model, kamera tahmini, NLP düzeltici veya kullanıcı
> arayüzü yoktur.

## Kapsam

İlk sürüm **Türk İşaret Dili çevirmeni değildir**. Sistem, ASL parmak alfabesiyle
harf tanıma ve ardından Türkçe metin restorasyonu yapacaktır. İşaret dili
cümlelerini veya TİD dil bilgisini çevirdiği iddia edilmez.

Planlanan veri akışı:

```text
Kamera -> El tespiti -> Görüntü ön işleme -> ASL sınıflandırıcı
       -> Tahmin kararlılığı/onay -> Ham metin -> Türkçe NLP -> Düzeltilmiş metin
```

Ayrıntılı kararlar için [mimari belgesine](docs/architecture.md), veri seti
bulguları için [veri seti analizine](docs/dataset_analysis.md), yerel bilgisayar
bulguları için [ortam raporuna](docs/environment_report.md) bakın.

## Veri seti

Ana aday [ASL Alphabet (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
veri setidir. Gerçek arşiv analizinde 200×200 RGB/JPEG biçiminde 87.000 eğitim
görseli, 29 sınıf ve 28 test dosyası bulunmuştur. Yayıncı test dosyalarının tamamı
eğitim dosyalarıyla kesin kopyadır; bağımsız test olarak kullanılmaz. Veri seti
depoya eklenmez. Lisans etiketi Kaggle üzerinde **GPL 2** olarak görünmektedir;
kullanım ve dağıtım öncesi güncel koşullar yeniden kontrol edilmelidir.

### İndirme

Kaggle hesabında API anahtarı oluşturulduktan sonra:

```powershell
python -m pip install kaggle
kaggle datasets download -d grassknoted/asl-alphabet -p data/raw
Expand-Archive -LiteralPath data/raw/asl-alphabet.zip -DestinationPath data/raw/asl-alphabet
```

`kaggle.json` gizli tutulmalı ve Git'e eklenmemelidir. İndirme yaklaşık 1,03 GiB
olduğundan sıkıştırılmış dosya ve açılmış veri için yeterli boş disk alanı
ayrılmalıdır. İndirmeden sonra Aşama 2'de dosya bütünlüğü, gerçek sınıf sayıları
ve görüntü özellikleri yerel olarak doğrulanacaktır.

## Geliştirme ortamı

Önerilen ve mevcut ortam Python 3.12 (64 bit), Windows 11'dir. İzole ortam:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` uyumluluk aralıklarını tanımlar. Aşama 2'nin hafif test ortamında
NumPy 2.5.5 ve Pillow 12.3.0 kurulmuştur; TensorFlow dahil tam eğitim ortamı henüz
kurulmamış ve kilit dosyası üretilmemiştir. Aşama 3 temiz kurulumundan sonra kesin
sürümler kaydedilecektir.

## Planlanan komutlar

Bu komutlar ilgili aşamalar tamamlandığında kullanılacaktır:

```powershell
# Eğitim (Aşama 3)
python -m recognition.train

# Değerlendirme (Aşama 3)
python -m recognition.evaluate

# Uygulama (Aşama 7)
streamlit run app.py

# Testler
python -m pytest
```

Mevcut yapısal, veri yükleme ve ön işleme testleri:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Bilinen sınırlamalar

- Veri seti indirilip analiz edildi; model henüz eğitilmedi ve doğruluk/FPS/gecikme
  sonucu yoktur.
- Veri setindeki `J` ve `Z` işaretleri hareketlidir; tek kare sınıflandırıcı hareketi
  öğrenemez. Statik görüntü başarısı gerçek hareket tanıma başarısı sayılmayacaktır.
- Yayıncı test klasöründe 28 görüntü vardır ve tamamı eğitim görüntülerinin kesin
  kopyasıdır; bağımsız test olarak kullanılamaz.
- Benzer çekimlerin rastgele bölünmesi veri sızıntısı ve iyimser metrik üretebilir.
- Kaggle görselleri ile gerçek kamera ortamı arasında arka plan, ışık ve el konumu
  farkından kaynaklanan alan kayması beklenir.
- Türkçe NLP modeli Aşama 6'da lisans, doğruluk, CPU gecikmesi ve bellek açısından
  deneysel olarak seçilecektir.

