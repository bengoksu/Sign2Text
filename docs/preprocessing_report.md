# Görüntü Ön İşleme Raporu

## Girdi sözleşmesi

Ortak ön işleme kodu `recognition/preprocessing.py` içindedir ve eğitim ile
ilerideki canlı kamera tahmininde aynı temel fonksiyon kullanılacaktır.

1. Dosya Pillow ile güvenli biçimde açılır ve tamamen decode edilir.
2. EXIF yönü uygulanır.
3. Gri/RGBA gibi girişler üç kanallı RGB'ye çevrilir.
4. En-boy oranı korunarak hedef tuvale yerleştirilir; kalan alan padding olur.
5. Varsayılan hedef `(224, 224, 3)` olarak seçilmiştir.
6. Varsayılan çıktı `float32` ve `[0, 1]` aralığındadır.

224×224 boyut, Aşama 3'te özel CNN ile MobileNet/EfficientNet sınıfı transfer
öğrenme modellerini aynı veri hattında karşılaştırmayı kolaylaştırır. Kaynak
görüntüler kare olduğu için bu veri setinde padding oluşmaz; fonksiyon gerçek
kamera kırpmalarındaki farklı en-boy oranlarını da güvenli yönetir.

Transfer öğrenme modeli farklı bir normalizasyon beklerse o dönüşüm modelin içinde
veya açık bir modele özgü katmanda uygulanacaktır. `[0,1]` normalizasyonu ile
ImageNet `preprocess_input` aynı anda uygulanmayacaktır.

## Veri artırma

Yalnız eğitim batch'lerinde, tohumlanabilir olarak şunlar uygulanır:

- parlaklık: 0,85-1,15,
- kontrast: 0,85-1,15,
- döndürme: en fazla ±10 derece,
- ölçek: 0,90-1,10,
- yatay/dikey konum değişimi: en fazla görüntünün %5'i.

Doğrulama ve test verisine artırma uygulanmaz. Yatay çevirme özellikle devre dışıdır;
sağ/sol el yönünü ve bazı işaretlerin geometrisini değiştirerek gerçekçi olmayan
veya anlamı belirsiz örnek üretebilir.

## Bellek ve batch yönetimi

`recognition.dataset_loader.iter_image_batches` görüntüleri yalnız ihtiyaç anında
diskten okur; tüm veri RAM'e yüklenmez. Yol indeksleri kullanılır, varsayılan batch
boyutu 32'dir ve karıştırma sabit seed ile tekrarlanabilir.

`build_tf_dataset` Aşama 3 tam TensorFlow ortamı için şu hattı hazırlar:

```text
file paths -> decode JPEG -> resize_with_pad -> normalize
           -> batch -> prefetch(AUTOTUNE)
```

Eğitimde shuffle ve hafif parlaklık/kontrast artırması bulunur. RAM cache varsayılan
olarak kullanılmaz; 87 bin görüntü ve 15,68 GiB RAM için gereksiz risk oluşturur.
Gerekirse yalnız önbellek profili ölçüldükten sonra disk cache değerlendirilecektir.

## Etiket ve bölme sözleşmesi

`config/label_mapping.json`, A-Z için 0-25, `del=26`, `nothing=27`, `space=28`
eşlemesini saklar. Bölmeler görüntü kopyalamaz; JSONL dosyalarında proje-göreli
yol, etiket, sayısal etiket ve sıra grubu bulunur.

## Test kapsamı

Sentetik görüntülerle:

- görüntü okuma ve bozuk dosya hatası,
- RGB dönüşümü,
- çıktı boyutu ve veri aralığı,
- deterministik artırma,
- etiket eşlemesi,
- batch boyutları,
- yol ve sıra grubu kesişimsizliği

test edilmiştir. Gerçek veri setinin tamamı ayrıca decode edilmiş; bozuk dosya
bulunmamıştır. TensorFlow bu hafif Aşama 2 ortamına kurulmadığından `tf.data`
fonksiyonunun gerçek çalışma testi Aşama 3 ortam kurulumuna bırakılmıştır; NumPy
batch yükleyici gerçek dosyalarda test edilmiştir.

