# Model Değerlendirme Kaydı

Bu dosya Aşama 3 ve sonraki deneylerin ölçülmüş sonuçları için ayrılmıştır.

Henüz veri seti indirilmedi, model eğitilmedi ve herhangi bir başarı, FPS, gecikme,
CER veya WER değeri ölçülmedi. Bu nedenle raporlanabilir model sonucu yoktur.

İleride her deney için en az şu bilgiler kaydedilecektir:

- veri seti sürümü, dosya manifesti ve bölme stratejisi,
- rastgelelik tohumları ve eğitim yapılandırması,
- accuracy, sınıf bazlı ve makro precision/recall/F1,
- confusion matrix ve özellikle `J`, `Z`, `space`, `del`, `nothing` sonuçları,
- bağımsız gerçek kamera testi,
- model boyutu, cihaz bilgisi, ısınma sonrası CPU gecikmesi ve FPS,
- NLP için CER, WER, diakritik restorasyon başarısı ve yanlış düzeltme oranı.

Eğitim/doğrulama sonuçları ile bağımsız kamera sonuçları ayrı tablolar halinde
sunulacaktır.

