# ASL Alphabet Veri Seti Analizi

**Analiz tarihi:** 18 Kasım 2026  
**Kaggle kimliği:** `grassknoted/asl-alphabet`  
**Sürüm:** 1 (22 Nisan 2018)  
**DOI:** `10.34740/KAGGLE/DSV/29550`

## İndirme ve bütünlük

Veri seti Kaggle'ın herkese açık API indirme uç noktasından kimlik doğrulaması
gerektirmeden `data/raw/asl-alphabet.zip` konumuna indirildi. Büyük dosya ve açılmış
görüntüler `.gitignore` kapsamındadır.

| Kontrol | Ölçülen değer | Sonuç |
|---|---:|---|
| ZIP boyutu | 1.100.887.034 bayt | Kaggle depolama başlığıyla aynı |
| ZIP MD5 | `9a0a397aef38a4a36209c5e1c4d1d837` | Kaggle ETag/MD5 ile aynı |
| ZIP SHA-256 | `7c572f14fbaff94f98835cfe71c7582dd379a5176e7c4f83dbf3a30e4b3f68c4` | Yerel kayıt |
| ZIP girdisi | 87.028 | Okunabilir |
| Güvensiz arşiv yolu | 0 | Yol taşması görülmedi |
| Açılmış görüntü baytı | 1.106.471.482 | Kaggle API metadatasıyla aynı |

Kaynaklar: [Kaggle veri seti sayfası](https://www.kaggle.com/datasets/grassknoted/asl-alphabet),
[Kaggle API kaydı](https://www.kaggle.com/api/v1/datasets/view/grassknoted/asl-alphabet).

Kaggle metadatasındaki lisans etiketi **GPL 2**'dir. Yeniden dağıtım öncesi güncel
Kaggle koşulları ve lisans yükümlülükleri yeniden incelenmelidir. Akademik raporda
veri seti ve DOI kaynak gösterilecektir. Bu not hukuki görüş değildir.

## Gerçek dosya yapısı

Arşiv çift katmanlı klasör adları kullanır:

```text
asl_alphabet_train/asl_alphabet_train/<class>/<image>.jpg
asl_alphabet_test/asl_alphabet_test/<label>_test.jpg
```

Tüm dosyalar Pillow ile gerçekten açılmış, decode edilmiş ve içerik hash'leri
hesaplanmıştır.

| Özellik | Gerçek sonuç |
|---|---:|
| Toplam görüntü | 87.028 |
| Eğitim kaynağı | 87.000 |
| Yayıncı test klasörü | 28 |
| Eğitim sınıfı | 29 |
| Her sınıftaki eğitim görüntüsü | 3.000 |
| Format | 87.028 JPEG |
| Boyut | 87.028 adet 200×200 |
| Renk modu | 87.028 RGB |
| Bozuk/okunamayan dosya | 0 |

### Sınıf dağılımı

| Sınıflar | Sınıf başına | Toplam |
|---|---:|---:|
| A-Z | 3.000 | 78.000 |
| `del` | 3.000 | 3.000 |
| `nothing` | 3.000 | 3.000 |
| `space` | 3.000 | 3.000 |
| **Toplam** |  | **87.000** |

Eğitim kaynağı sınıf sayısı açısından tam dengelidir.

## Test klasörü ve veri sızıntısı bulgusu

Yayıncı açıklamasında 29 test görüntüsü yazmasına rağmen arşivde **28** görüntü
vardır; `del_test.jpg` eksiktir. Daha önemlisi, 28 test görüntüsünün tamamı eğitim
klasöründeki bir görüntüyle SHA-256 düzeyinde aynıdır. Tam veri setinde 28 kesin
kopya grubu ve 28 fazla kopya dosya bulunmuştur; bunlar train/test klasörleri
arasındadır.

Bu nedenle yayıncı test klasörü:

- bağımsız test kümesi değildir,
- accuracy/F1 gibi genelleme metrikleri için kullanılmayacaktır,
- eğitim veya doğrulamaya katılmayacaktır,
- yalnız dosya okuma/tahmin hattında tanısal örnek olarak tutulacaktır.

## Benzerlik analizi

Tam O(n²) çift karşılaştırması yerine tüm dosyalarda 64-bit dHash üretildi; aynı
sınıftaki eşit dHash grupları ve sayısal olarak ardışık dosya çiftleri incelendi.

| Ölçüm | Sonuç |
|---|---:|
| Aynı sınıfta eşit dHash aday grubu | 13.381 |
| Bu gruplardaki fazla aday görüntü | 26.837 |
| Ardışık sayısal dosya çifti | 86.971 |
| Ardışık çiftlerde medyan dHash uzaklığı | 1 / 64 bit |
| Hamming uzaklığı ≤2 olan ardışık çift | 63.626 (%73,16) |
| Hamming uzaklığı ≤5 olan ardışık çift | 83.539 (%96,05) |

dHash eşitliği tek başına byte düzeyinde kopya anlamına gelmez; bunlar görsel
benzerlik adaylarıdır. Sonuçlar veri setinin ardışık, birbirine çok benzeyen
karelerden oluştuğunu güçlü biçimde gösterir. Rastgele görüntü bölmesi yapay olarak
iyimser sonuç üretme riski taşır.

## Bölme stratejisi

Veri setinde kişi veya çekim oturumu kimliği yoktur. Bu nedenle kişi-bağımsız
değerlendirme yapılamaz. Dosya adlarındaki sayısal sıra kullanılarak her sınıfta
50 ardışık görüntü tek bir grup kabul edilmiştir. Gruplar sabit `seed=42` ile
%70/%15/%15 oranında ayrılmıştır; bir grup iki bölüme giremez.

| Bölüm | Toplam | Sınıf başına | Oran |
|---|---:|---:|---:|
| Eğitim | 60.900 | 2.100 | %70 |
| Doğrulama | 13.050 | 450 | %15 |
| İç test | 13.050 | 450 | %15 |

Yol ve grup kesişimi gerçek indekslerde **0** olarak doğrulanmıştır. Görüntüler
kopyalanmamış, `data/processed/splits/*.jsonl` yol indeksleri oluşturulmuştur.

Bu yaklaşım komşu kare sızıntısını azaltır ancak aynı kişi/arka planın farklı
gruplarda bulunmasını engellediğini kanıtlamaz. Aşama 3 sonuçları “kişi bağımsız”
olarak sunulmayacak; gerçek genelleme için proje ekibinin ayrı kamera veri seti
gerekecektir.

## J ve Z sınırlaması

ASL'de `J` ve `Z` hareket yörüngesi içerir. Bu veri setindeki tek kareler hareketin
tamamını temsil etmez. Statik CNN bu sınıflarda yalnız görüntüdeki anlık poz veya
veri setine özgü ipuçlarını öğrenebilir; gerçek hareket tanıma iddiası yapılamaz.

## Tekrarlanabilir analiz

Ham ayrıntılı sonuç Git dışında tutulan
`data/processed/dataset_analysis.json` dosyasındadır. Analiz şu komutla yeniden
üretilebilir:

```powershell
.venv\Scripts\python.exe -m recognition.analyze_dataset --workers 8
```

