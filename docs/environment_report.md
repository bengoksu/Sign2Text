# Geliştirme Ortamı Raporu

**Kontrol tarihi:** 18 Kasım 2026  
**Çalışma dizini:** `C:\Users\HP\Desktop\Sign2Text`

| Bileşen | Bulgular |
|---|---|
| İşletim sistemi | Windows 11 Home Single Language, 64 bit |
| PowerShell | 5.1.26100.9818 |
| Python | 3.12.6 (`...\Python312\python.exe`) |
| pip | 24.2 |
| Git | 2.52.0.windows.1 |
| CPU | Intel Core i5-12500H, 12 çekirdek / 16 mantıksal işlemci |
| RAM | 15,68 GiB |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB VRAM |
| NVIDIA sürücüsü | 610.91 |
| C: boş alan | yaklaşık 20,38 GiB |
| Kaggle CLI | Kurulu değil |
| FFmpeg | Kurulu değil |
| Python paketleri | Global yorumlayıcıda yalnız `pip` listelendi |
| Git deposu | Kontrol başlangıcında yoktu; Aşama 1'de başlatıldı |

## Değerlendirme

Python 3.12 için TensorFlow, MediaPipe, OpenCV, Streamlit, Transformers ve PyTorch
paket sürümleri ile Windows wheel metadataları PyPI dizininde mevcuttur. Tam
bağımlılık çözümleme denemesi paket kurmadan başlatılmış, uzun sürdüğü için
durdurulmuştur. Paketlerin birlikte temiz kurulum testi henüz yapılmamıştır;
`requirements.txt` bu nedenle kesin kilit değil, uyumluluk aralıkları içerir.

Windows üzerinde TensorFlow'un güncel sürümleri CPU ile çalışabilir. NVIDIA GPU ile
eğitim planlanırsa güncel TensorFlow yönergelerine göre WSL2 yolu ayrıca
değerlendirilmelidir; yerel Windows GPU desteği varsayılmamıştır. İlk hedef CPU'da
çalışabilirliktir.

Global Python ortamına bağımlılık kurulmayacaktır. `.venv` oluşturulması ve temiz
kurulum/ithalat testi Aşama 2'nin başlangıç koşuludur. Yaklaşık 20 GiB boş alan veri
seti, açılmış kopya, ön işlenmiş veriler, paket önbelleği ve modeller için dar
olabilir; indirme öncesinde alan izlenmelidir.

