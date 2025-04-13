# Soruların Cevaplarını Bulmak için QA Sistemi Oluşturma
QA sistemimiz için ilk ihtiyacımız, bir müşteri yorumundaki potansiyel bir cevabı metin aralığı olarak tanımlamanın bir yolunu bulmaktır. Örneğin, "Su geçirmez mi?" gibi bir sorumuz varsa ve yorum pasajı "Bu saat 30m derinlikte su geçirmez" ise, model "30m'de su geçirmez" çıktısını vermelidir. Bunu yapmak için şunları anlamamız gerekir:
• Denetimli öğrenme problemini çerçevelemek.
• QA görevleri için metni tokenize etmek ve kodlamak.
• Bir modelin maksimum bağlam boyutunu aşan uzun pasajlarla başa çıkmak.
Sorunu nasıl çerçeveleyeceğimize bakalım.

## Span Sınıflandırması
Metinden cevapları çıkarmak için en yaygın yöntem, sorunun başlangıç ve bitiş tokenlerini bir modelin tahmin etmesi gereken etiketler olarak ele alan bir span sınıflandırma görevi olarak çerçevelemektir. Bu işlem Şekil 7-4'te gösterilmektedir.
### Şekil 7-4. QA görevleri için span sınıflandırma başlığı
Eğitim setimiz nispeten küçük olduğundan, sadece 1.295 örnek içerdiğinden, iyi bir strateji, önceden SQuAD gibi büyük ölçekli bir QA veri kümesinde ince ayar yapılmış bir dil modeliyle başlamaktır. Genel olarak, bu modeller güçlü okuma anlama yeteneklerine sahiptir ve daha doğru bir sistem oluşturmak için iyi bir temel oluşturur. Bu, önceki bölümlerde aldığımız yaklaşımdan biraz farklıdır, burada genellikle önceden eğitilmiş bir modelle başlıyor ve görev-specifik başlığı kendi başımıza inceliyorduk. Örneğin, Bölüm 2'de, sınıf sayısının eldeki veri kümesine bağlı olması nedeniyle sınıflandırma başlığını ince ayar yapmak zorunda kaldık. Extractive QA için, aslında etiketlerin yapısı veri kümeleri arasında aynı kaldığından ince ayar yapılmış bir modelle başlayabiliriz.

### Hugging Face Hub'da Extractive QA Modelleri
Model listesindeki "squad" araması yaparak extractive QA modellerini bulabilirsiniz (Şekil 7-5).
#### Şekil 7-5. Hugging Face Hub'daki extractive QA modellerinden bir seçki
Görüldüğü gibi, yazı yazıldığı sırada seçmek için 350'den fazla QA modeli bulunmaktadır - hangisini seçmelisiniz? Genel olarak, cevap çeşitli faktörlere bağlıdır, örneğin korpusunuzun mono- veya çok dilli olup olmadığı ve modeli bir üretim ortamında çalıştırma kısıtlamaları. Tablo 7-2, üzerine inşa etmek için iyi bir temel sağlayan birkaç modeli listelemektedir.

### Tablo 7-2. SQuAD 2.0'da ince ayar yapılmış temel transformer modelleri
Bu bölümün amaçları doğrultusunda, hızlı bir şekilde eğitilebildiği ve keşfedeceğimiz teknikler üzerinde hızlı bir şekilde yineleme yapmamızı sağladığı için ince ayar yapılmış bir MiniLM modeli kullanacağız. İlk olarak, metinlerimizi kodlamak için bir tokenize ediciye ihtiyacımız var, bu nedenle QA görevleri için bunun nasıl çalıştığına bakalım.

## QA için Metni Tokenize Etme
Metinlerimizi kodlamak için MiniLM model kontrol noktasını yükleyeceğiz:
```python
from transformers import AutoTokenizer
model_ckpt = "deepset/minilm-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
```
Modeli eylemde görmek için, önce kısa bir metin pasajından bir cevap çıkarmaya çalışalım. Extractive QA görevlerinde, girdiler (soru, bağlam) çiftleri olarak sağlanır, bu nedenle her ikisini de tokenize ediciye aşağıdaki gibi geçiriyoruz:
```python
question = "Bu ne kadar müzik tutabilir?"
context = """Bir MP3 yaklaşık 1 MB/dakikadır, bu nedenle dosya boyutuna bağlı olarak yaklaşık 6000 saat."""
inputs = tokenizer(question, context, return_tensors="pt")
```
Burada PyTorch Tensor nesneleri döndürdük, çünkü modeli ileri besleme için bunlara ihtiyacımız olacak. Tokenize edilmiş girdileri bir tablo olarak görüntülediğimizde:
```python
print(tokenizer.decode(inputs["input_ids"][0]))
```
çıktı:
```
[CLS] bu ne kadar müzik tutabilir? [SEP] bir mp3 yaklaşık 1 mb / dakika, bu nedenle yaklaşık 6000 saat dosya boyutuna bağlı olarak. [SEP]
```
Gördüğümüz gibi, her QA örneği için girdiler şu biçimi alır:
```
[CLS] soru tokenleri [SEP] bağlam tokenleri [SEP]
```
burada ilk [SEP] tokeninin konumu token_type_ids tarafından belirlenir. Şimdi metnimizi tokenize ettiğimize göre, sadece bir QA başlığı olan bir model oluşturmamız ve girdileri ileri besleme yoluyla çalıştırmamız gerekiyor:
```python
import torch
from transformers import AutoModelForQuestionAnswering
model = AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
with torch.no_grad():
    outputs = model(**inputs)
print(outputs)
```
Çıktı:
```
QuestionAnsweringModelOutput(loss=None, start_logits=tensor([[-0.9862, -4.7750,  -5.4025, -5.2378, -5.2863, -5.5117, -4.9819, -6.1880,
         -0.9862,  0.2596, -0.2144, -1.7136,  3.7806,  4.8561, -1.0546, -3.9097,
         -1.7374, -4.5944, -1.4278,  3.9949,  5.0390, -0.2018, -3.0193, -4.8549,
         -2.3107, -3.5110, -3.5713, -0.9862]]), end_logits=tensor([[-0.9623,
         -5.4733, -5.0326, -5.1639, -5.4278, -5.5151, -5.1749, -4.6233,
         -0.9623, -3.7855, -0.8715, -3.7745, -3.0161, -1.1780,  0.1758, -2.7365,
          4.8934,  0.3046, -3.1761, -3.2762,  0.8937,  5.6606, -0.3623, -4.9554,
         -3.2531, -0.0914,  1.6211, -0.9623]]), hidden_states=None, 
attentions=None)
```
Burada QA başlığının çıktısı olarak bir QuestionAnsweringModelOutput nesnesi aldığımızı görebiliriz. Şekil 7-4'te gösterildiği gibi, QA başlığı, kodlayıcıdan gizli durumları alan ve başlangıç ve bitiş aralıkları için logitleri hesaplayan doğrusal bir katmandır. Bu, QA'yı Bölüm 4'te karşılaştığımıza benzer bir şekilde token sınıflandırması olarak ele aldığımız anlamına gelir.

## Cevap Aralığını Çıkarma
Cevap aralığını çıkarmak için önce başlangıç ve bitiş tokenleri için logitleri almamız gerekir:
```python
start_logits = outputs.start_logits
end_logits = outputs.end_logits
```
Bu logitlerin şekillerini input ID'leri ile karşılaştırdığımızda:
```python
print(f"Input IDs şekli: {inputs.input_ids.size()}")
print(f"Başlangıç logitleri şekli: {start_logits.size()}")
print(f"Bitiş logitleri şekli: {end_logits.size()}")
```
çıktı:
```
Input IDs şekli: torch.Size([1, 28])
Başlangıç logitleri şekli: torch.Size([1, 28])
Bitiş logitleri şekli: torch.Size([1, 28])
```
Her input tokeni için iki logit (başlangıç ve bitiş) olduğunu görebiliriz. Şekil 7-6'da gösterildiği gibi, daha büyük, pozitif logitler başlangıç ve bitiş tokenlerinin daha olası adaylarına karşılık gelir.

## Nihai Cevabı Alma
Nihai cevabı almak için, başlangıç ve bitiş token logitleri üzerinde argmax hesaplayabilir ve sonra girdilerden aralığı dilimleyebiliriz. Aşağıdaki kod bu adımları gerçekleştirir ve ortaya çıkan metni yazdırır:
```python
import torch
start_idx = torch.argmax(start_logits)
end_idx = torch.argmax(end_logits) + 1
answer_span = inputs["input_ids"][0][start_idx:end_idx]
answer = tokenizer.decode(answer_span)
print(f"Soru: {question}")
print(f"Cevap: {answer}")
```
Çıktı:
```
Soru: Bu ne kadar müzik tutabilir?
Cevap: 6000 saat
```
## Transformers Pipeline Kullanma
Tüm bu ön işleme ve son işleme adımları, Transformers'da bir dedicated pipeline içinde düzgün bir şekilde paketlenmiştir. Pipeline'ı tokenize edici ve ince ayar yapılmış model ile aşağıdaki gibi başlatabiliriz:
```python
from transformers import pipeline
pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
pipe(question=question, context=context, topk=3)
```
Çıktı:
```
[{'score': 0.26516005396842957,
  'start': 38,
  'end': 48,
  'answer': '6000 saat'},
 {'score': 0.2208300083875656,
  'start': 16,
  'end': 48,
  'answer': '1 MB/dakika, bu nedenle yaklaşık 6000 saat'},
 {'score': 0.10253632068634033,
  'start': 16,
  'end': 27,
  'answer': '1 MB/dakika'}]
```
## Uzun Pasajlarla Başa Çıkma
Okuma anlama modellerinin karşılaştığı bir incelik, bağlamın genellikle modelin maksimum dizi uzunluğundan daha fazla token içermesidir. Şekil 7-7'de gösterildiği gibi, SubjQA eğitim setinin önemli bir kısmı, MiniLM'in bağlam boyutuna sığmayan soru-bağlam çiftleri içerir.

### Kaydırmalı Pencere Kullanma
Diğer görevler için, metin sınıflandırması gibi, uzun metinleri basitçe kırparak yeterli bilginin [CLS] tokeninin embeddinginde bulunduğunu varsaydık. Ancak QA için bu strateji sorunludur çünkü bir sorunun cevabı bağlamın sonuna yakın olabilir ve bu nedenle kırpma tarafından kaldırılabilir. Şekil 7-8'de gösterildiği gibi, bununla başa çıkmak için standart yöntem, girdilere kaydırmalı bir pencere uygulamaktır.

### Transformers'da Kaydırmalı Pencere
Transformers'da, tokenize edicide return_overflowing_tokens=True ayarlayarak kaydırmalı pencereyi etkinleştirebiliriz. Kaydırmalı pencerenin boyutu max_seq_length argümanı tarafından kontrol edilir ve adım boyutu doc_stride tarafından kontrol edilir. Eğitim setimizden ilk örneği alalım ve bunun nasıl çalıştığını göstermek için küçük bir pencere tanımlayalım:
```python
example = dfs["train"].iloc[0][["question", "context"]]
tokenized_example = tokenizer(example["question"], example["context"],
                               return_overflowing_tokens=True, max_length=100,
                               stride=25)
```
Bu durumda, her pencere için bir input_ids listesi alırız. Her pencerede kaç token olduğunu kontrol edelim:
```python
for idx, window in enumerate(tokenized_example["input_ids"]):
    print(f"Pencere #{idx} {len(window)} token içerir")
```
Çıktı:
```
Pencere #0 100 token içerir
Pencere #1 88 token içerir
```
İki pencerenin nerede örtüştüğünü görmek için girdileri çözebiliriz:
```python
for window in tokenized_example["input_ids"]:
    print(f"{tokenizer.decode(window)}\n")
```
Çıktı:
```
[CLS] bas nasıl? [SEP] koss kulaklıklarını geçmişte, pro 4aa ve qz - 99'u denedim. koss portapro taşınabilir ve harika bas cevabı var. android telefonumla iyi çalışıyor ve "yuvarlanarak" motosiklet ceketimde veya bilgisayar çantamda taşınabilir. çok hafif ve kulaklarınıza ağırlık yapmıyor, hatta tüm gün müzik dinleseniz bile. ses, herhangi bir kulaklıkla kıyaslanamayacak kadar iyi ve pro 4aa kadar iyi. "açık hava" kulaklıkları olduğundan, kapalı tiplerle eşleşen bas sesini yakalar. 32$ için, yanlış gidemezsiniz. [SEP]

[CLS] bas nasıl? [SEP] ve kulaklarınıza ağırlık yapmıyor, hatta tüm gün müzik dinleseniz bile. ses, herhangi bir kulaklıkla kıyaslanamayacak kadar iyi ve pro 4aa kadar iyi. "açık hava" kulaklıkları olduğundan, kapalı tiplerle eşleşen bas sesini yakalar. 32$ için, yanlış gidemezsiniz. [SEP]
```

---

