# Transformer Encoder Katmanı

Daha önce gördüğümüz gibi, transformer'ın kodlayıcısı (encoder) birbirinin yanında dizilmiş birçok kodlayıcı katmandan oluşur. Şekil 3-2'de gösterildiği gibi, her bir kodlayıcı katmanı bir dizi gömme (embedding) alır ve bunları aşağıdaki alt katmanlardan geçirir:
* Çok başlı öz dikkat (self-attention) katmanı
* Her bir girdi gömmesine uygulanan tamamen bağlı ileri beslemeli (feed-forward) katman

Her bir kodlayıcı katmanının çıktı gömmeleri, girdilerle aynı boyuttadır ve yakında göreceğimiz gibi, kodlayıcı yığınının ana rolü, dizideki bazı bağlamsal bilgileri kodlayan temsiller üretmek için girdi gömmelerini "güncellemektir". Örneğin, "apple" kelimesi, eğer "keynote" veya "phone" kelimeleri ona yakınsa, daha "şirket benzeri" ve daha az "meyve benzeri" olacak şekilde güncellenir.
# Encoder Katmanına Yakından Bakış

Şekil 3-2. Encoder katmanına yakınlaştırma

Bu alt katmanların her biri aynı zamanda atlama bağlantıları (skip connections) ve katman normalizasyonu (layer normalization) kullanır, bunlar derin sinir ağlarını etkili bir şekilde eğitmek için standart numaralar. Ancak bir transformer'ın nasıl çalıştığını gerçekten anlamak için daha derine gitmemiz gerekiyor. En önemli yapı taşı olan öz dikkat katmanıyla başlayalım.

Paragrafta Python kodları bulunmamaktadır.

---

