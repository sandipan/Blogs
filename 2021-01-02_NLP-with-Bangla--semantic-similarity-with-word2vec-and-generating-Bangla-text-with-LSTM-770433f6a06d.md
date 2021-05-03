NLP with Bangla: semantic similarity with word2vec and generating Bangla text with LSTM {.p-name}
=======================================================================================

Training word2vec model with Bangla wiki corpus with tensorflow and
using t-SNE to visualize the semantic similarity between words, also…

* * * * *

### NLP with Bangla: semantic similarity with word2vec and generating Bangla text with LSTM {#2b1e .graf .graf--h3 .graf--leading .graf--title name="2b1e"}

#### Training word2vec model with Bangla wiki corpus with tensorflow and using t-SNE to visualize the semantic similarity between words, also training a character / word LSTM on selected Tagore’s songs to generate songs like Tagore with keras {#cdc0 .graf .graf--h4 .graf-after--h3 .graf--subtitle name="cdc0"}

**Word2vec model with Bangla wiki corpus with tensorflow**

-   Let’s start by importing the required libraries

``` {#43af .graf .graf--pre .graf-after--li name="43af"}
import collectionsimport numpy as npimport tensorflow as tffrom matplotlib import pylab
```

-   Download the Bangla wikipedia corpus from
    [Kaggle](https://www.kaggle.com/shazol/bangla-wikipedia-corpus). The
    first few lines from the corpus are shown below:

``` {#9607 .graf .graf--pre .graf-after--li name="9607"}
id,text,title,url1528,"রবীন্দ্রনাথ ঠাকুর
```

``` {#d08f .graf .graf--pre .graf-after--pre name="d08f"}
রবীন্দ্রনাথ ঠাকুর (৭ই মে, ১৮৬১ - ৭ই আগস্ট, ১৯৪১) (২৫ বৈশাখ, ১২৬৮ - ২২ শ্রাবণ, ১৩৪৮ বঙ্গাব্দ) ছিলেন অগ্রণী বাঙালি কবি, ঔপন্যাসিক, সংগীতস্রষ্টা, নাট্যকার, চিত্রকর, ছোটগল্পকার, প্রাবন্ধিক, অভিনেতা, কণ্ঠশিল্পী ও দার্শনিক। তাঁকে বাংলা ভাষার সর্বশ্রেষ্ঠ সাহিত্যিক মনে করা হয়। রবীন্দ্রনাথকে গুরুদেব, কবিগুরু ও বিশ্বকবি অভিধায় ভূষিত করা হয়। রবীন্দ্রনাথের ৫২টি কাব্যগ্রন্থ, ৩৮টি নাটক, ১৩টি উপন্যাস ও ৩৬টি প্রবন্ধ ও অন্যান্য গদ্যসংকলন তাঁর জীবদ্দশায় বা মৃত্যুর অব্যবহিত পরে প্রকাশিত হয়। তাঁর সর্বমোট ৯৫টি ছোটগল্প ও ১৯১৫টি গান যথাক্রমে ""গল্পগুচ্ছ"" ও ""গীতবিতান"" সংকলনের অন্তর্ভুক্ত হয়েছে। রবীন্দ্রনাথের যাবতীয় প্রকাশিত ও গ্রন্থাকারে অপ্রকাশিত রচনা ৩২ খণ্ডে ""রবীন্দ্র রচনাবলী"" নামে প্রকাশিত হয়েছে। রবীন্দ্রনাথের যাবতীয় পত্রসাহিত্য উনিশ খণ্ডে ""চিঠিপত্র"" ও চারটি পৃথক গ্রন্থে প্রকাশিত। এছাড়া তিনি প্রায় দুই হাজার ছবি এঁকেছিলেন। রবীন্দ্রনাথের রচনা বিশ্বের বিভিন্ন ভাষায় অনূদিত হয়েছে। ১৯১৩ সালে ""গীতাঞ্জলি"" কাব্যগ্রন্থের ইংরেজি অনুবাদের জন্য তিনি সাহিত্যে নোবেল পুরস্কার লাভ করেন।
```

``` {#3ad3 .graf .graf--pre .graf-after--pre name="3ad3"}
রবীন্দ্রনাথ ঠাকুর কলকাতার এক ধনাঢ্য ও সংস্কৃতিবান ব্রাহ্ম পিরালী ব্রাহ্মণ পরিবারে জন্মগ্রহণ করেন। বাল্যকালে প্রথাগত বিদ্যালয়-শিক্ষা তিনি গ্রহণ করেননি; গৃহশিক্ষক রেখে বাড়িতেই তাঁর শিক্ষার ব্যবস্থা করা হয়েছিল। আট বছর বয়সে তিনি কবিতা লেখা শুরু করেন। ১৮৭৪ সালে ""তত্ত্ববোধিনী পত্রিকা""-এ তাঁর """" কবিতাটি প্রকাশিত হয়। এটিই ছিল তাঁর প্রথম প্রকাশিত রচনা। ১৮৭৮ সালে মাত্র সতেরো বছর বয়সে রবীন্দ্রনাথ প্রথমবার ইংল্যান্ডে যান। ১৮৮৩ সালে মৃণালিনী দেবীর সঙ্গে তাঁর বিবাহ হয়। ১৮৯০ সাল থেকে রবীন্দ্রনাথ পূর্ববঙ্গের শিলাইদহের জমিদারি এস্টেটে বসবাস শুরু করেন। ১৯০১ সালে তিনি পশ্চিমবঙ্গের শান্তিনিকেতনে ব্রহ্মচর্যাশ্রম প্রতিষ্ঠা করেন এবং সেখানেই পাকাপাকিভাবে বসবাস শুরু করেন। ১৯০২ সালে তাঁর পত্নীবিয়োগ হয়। ১৯০৫ সালে তিনি বঙ্গভঙ্গ-বিরোধী আন্দোলনে জড়িয়ে পড়েন। ১৯১৫ সালে ব্রিটিশ সরকার তাঁকে নাইট উপাধিতে ভূষিত করেন। কিন্তু ১৯১৯ সালে জালিয়ানওয়ালাবাগ হত্যাকাণ্ডের প্রতিবাদে তিনি সেই উপাধি ত্যাগ করেন। ১৯২১ সালে গ্রামোন্নয়নের জন্য তিনি শ্রীনিকেতন নামে একটি সংস্থা প্রতিষ্ঠা করেন। ১৯২৩ সালে আনুষ্ঠানিকভাবে বিশ্বভারতী প্রতিষ্ঠিত হয়। দীর্ঘজীবনে তিনি বহুবার বিদেশ ভ্রমণ করেন এবং সমগ্র বিশ্বে বিশ্বভ্রাতৃত্বের বাণী প্রচার করেন। ১৯৪১ সালে দীর্ঘ রোগভোগের পর কলকাতার পৈত্রিক বাসভবনেই তাঁর মৃত্যু হয়।
```

``` {#860b .graf .graf--pre .graf-after--pre name="860b"}
রবীন্দ্রনাথের কাব্যসাহিত্যের বৈশিষ্ট্য ভাবগভীরতা, গীতিধর্মিতা চিত্ররূপময়তা, অধ্যাত্মচেতনা, ঐতিহ্যপ্রীতি, প্রকৃতিপ্রেম, মানবপ্রেম, স্বদেশপ্রেম, বিশ্বপ্রেম, রোম্যান্টিক সৌন্দর্যচেতনা, ভাব, ভাষা, ছন্দ ও আঙ্গিকের বৈচিত্র্য, বাস্তবচেতনা ও প্রগতিচেতনা। রবীন্দ্রনাথের গদ্যভাষাও কাব্যিক। ভারতের ধ্রুপদি ও লৌকিক সংস্কৃতি এবং পাশ্চাত্য বিজ্ঞানচেতনা ও শিল্পদর্শন তাঁর রচনায় গভীর প্রভাব বিস্তার করেছিল। কথাসাহিত্য ও প্রবন্ধের মাধ্যমে তিনি সমাজ, রাজনীতি ও রাষ্ট্রনীতি সম্পর্কে নিজ মতামত প্রকাশ করেছিলেন। সমাজকল্যাণের উপায় হিসেবে তিনি গ্রামোন্নয়ন ও গ্রামের দরিদ্র মানুষ কে শিক্ষিত করে তোলার পক্ষে মতপ্রকাশ করেন। এর পাশাপাশি সামাজিক ভেদাভেদ, অস্পৃশ্যতা, ধর্মীয় গোঁড়ামি ও ধর্মান্ধতার বিরুদ্ধেও তিনি তীব্র প্রতিবাদ জানিয়েছিলেন। রবীন্দ্রনাথের দর্শনচেতনায় ঈশ্বরের মূল হিসেবে মানব সংসারকেই নির্দিষ্ট করা হয়েছে; রবীন্দ্রনাথ দেববিগ্রহের পরিবর্তে কর্মী অর্থাৎ মানুষ ঈশ্বরের পূজার কথা বলেছিলেন। সংগীত ও নৃত্যকে তিনি শিক্ষার অপরিহার্য অঙ্গ মনে করতেন। রবীন্দ্রনাথের গান তাঁর অন্যতম শ্রেষ্ঠ কীর্তি। তাঁর রচিত ""আমার সোনার বাংলা"" ও ""জনগণমন-অধিনায়ক জয় হে"" গানদুটি যথাক্রমে গণপ্রজাতন্ত্রী বাংলাদেশ ও ভারতীয় প্রজাতন্ত্রের জাতীয় সংগীত।
```

``` {#09cf .graf .graf--pre .graf-after--pre name="09cf"}
জীবন.প্রথম জীবন (১৮৬১–১৯০১).শৈশব ও কৈশোর (১৮৬১ - ১৮৭৮).রবীন্দ্রনাথ ঠাকুর কলকাতার জোড়াসাঁকো ঠাকুরবাড়িতে জন্মগ্রহণ করেছিলেন। তাঁর পিতা ছিলেন ব্রাহ্ম ধর্মগুরু দেবেন্দ্রনাথ ঠাকুর (১৮১৭–১৯০৫) এবং মাতা ছিলেন সারদাসুন্দরী দেবী (১৮২৬–১৮৭৫)। রবীন্দ্রনাথ ছিলেন পিতামাতার চতুর্দশ সন্তান। জোড়াসাঁকোর ঠাকুর পরিবার ছিল ব্রাহ্ম আদিধর্ম মতবাদের প্রবক্তা। রবীন্দ্রনাথের পূর্ব পুরুষেরা খুলনা জেলার রূপসা উপজেলা পিঠাভোগে বাস করতেন। ১৮৭৫ সালে মাত্র চোদ্দ বছর বয়সে রবীন্দ্রনাথের মাতৃবিয়োগ ঘটে। পিতা দেবেন্দ্রনাথ দেশভ্রমণের নেশায় বছরের অধিকাংশ সময় কলকাতার বাইরে অতিবাহিত করতেন। তাই ধনাঢ্য পরিবারের সন্তান হয়েও রবীন্দ্রনাথের ছেলেবেলা কেটেছিল ভৃত্যদের অনুশাসনে। শৈশবে রবীন্দ্রনাথ কলকাতার ওরিয়েন্টাল সেমিনারি, নর্ম্যাল স্কুল, বেঙ্গল অ্যাকাডেমি এবং সেন্ট জেভিয়ার্স কলেজিয়েট স্কুলে কিছুদিন করে পড়াশোনা করেছিলেন। কিন্তু বিদ্যালয়-শিক্ষায় অনাগ্রহী হওয়ায় বাড়িতেই গৃহশিক্ষক রেখে তাঁর শিক্ষার ব্যবস্থা করা হয়েছিল। ছেলেবেলায় জোড়াসাঁকোর বাড়িতে অথবা বোলপুর ও পানিহাটির বাগানবাড়িতে প্রাকৃতিক পরিবেশের মধ্যে ঘুরে বেড়াতে বেশি স্বচ্ছন্দবোধ করতেন রবীন্দ্রনাথ।
```

``` {#9466 .graf .graf--pre .graf-after--pre name="9466"}
১৮৭৩ সালে এগারো বছর বয়সে রবীন্দ্রনাথের উপনয়ন অনুষ্ঠিত হয়েছিল। এরপর তিনি কয়েক মাসের জন্য পিতার সঙ্গে দেশভ্রমণে বের হন। প্রথমে তাঁরা আসেন শান্তিনিকেতনে। এরপর পাঞ্জাবের অমৃতসরে কিছুকাল কাটিয়ে শিখদের উপাসনা পদ্ধতি পরিদর্শন করেন। শেষে পুত্রকে নিয়ে দেবেন্দ্রনাথ যান পাঞ্জাবেরই (অধুনা ভারতের হিমাচল প্রদেশ রাজ্যে অবস্থিত) ডালহৌসি শৈলশহরের নিকট বক্রোটায়। এখানকার বক্রোটা বাংলোয় বসে রবীন্দ্রনাথ পিতার কাছ থেকে সংস্কৃত ব্যাকরণ, ইংরেজি, জ্যোতির্বিজ্ঞান, সাধারণ বিজ্ঞান ও ইতিহাসের নিয়মিত পাঠ নিতে শুরু করেন। দেবেন্দ্রনাথ তাঁকে বিশিষ্ট ব্যক্তিবর্গের জীবনী, কালিদাস রচিত ধ্রুপদি সংস্কৃত কাব্য ও নাটক এবং উপনিষদ্‌ পাঠেও উৎসাহিত করতেন। ১৮৭৭ সালে ""ভারতী"" পত্রিকায় তরুণ রবীন্দ্রনাথের কয়েকটি গুরুত্বপূর্ণ রচনা প্রকাশিত হয়। এগুলি হল মাইকেল মধুসূদনের """", ""ভানুসিংহ ঠাকুরের পদাবলী"" এবং """" ও """" নামে দুটি গল্প। এর মধ্যে ""ভানুসিংহ ঠাকুরের পদাবলী"" বিশেষভাবে উল্লেখযোগ্য। এই কবিতাগুলি রাধা-কৃষ্ণ বিষয়ক পদাবলির অনুকরণে ""ভানুসিংহ"" ভণিতায় রচিত। রবীন্দ্রনাথের ""ভিখারিণী"" গল্পটি (১৮৭৭) বাংলা সাহিত্যের প্রথম ছোটগল্প। ১৮৭৮ সালে প্রকাশিত হয় রবীন্দ্রনাথের প্রথম কাব্যগ্রন্থ তথা প্রথম মুদ্রিত গ্রন্থ ""কবিকাহিনী""। এছাড়া এই পর্বে তিনি রচনা করেছিলেন """" (১৮৮২) কাব্যগ্রন্থটি। রবীন্দ্রনাথের বিখ্যাত কবিতা """" এই কাব্যগ্রন্থের অন্তর্গত।
```

``` {#a74f .graf .graf--pre .graf-after--pre name="a74f"}
যৌবন (১৮৭৮-১৯০১).১৮৭৮ সালে ব্যারিস্টারি পড়ার উদ্দেশ্যে ইংল্যান্ডে যান রবীন্দ্রনাথ। প্রথমে তিনি ব্রাইটনের একটি পাবলিক স্কুলে ভর্তি হয়েছিলেন। ১৮৭৯ সালে ইউনিভার্সিটি কলেজ লন্ডনে আইনবিদ্যা নিয়ে পড়াশোনা শুরু করেন। কিন্তু সাহিত্যচর্চার আকর্ষণে সেই পড়াশোনা তিনি সমাপ্ত করতে পারেননি। ইংল্যান্ডে থাকাকালীন শেকসপিয়র ও অন্যান্য ইংরেজ সাহিত্যিকদের রচনার সঙ্গে রবীন্দ্রনাথের পরিচয় ঘটে। এই সময় তিনি বিশেষ মনোযোগ সহকারে পাঠ করেন ""রিলিজিও মেদিচি"", ""কোরিওলেনাস"" এবং ""অ্যান্টনি অ্যান্ড ক্লিওপেট্রা""। এই সময় তাঁর ইংল্যান্ডবাসের অভিজ্ঞতার কথা ""ভারতী"" পত্রিকায় পত্রাকারে পাঠাতেন রবীন্দ্রনাথ। উক্ত পত্রিকায় এই লেখাগুলি জ্যেষ্ঠভ্রাতা দ্বিজেন্দ্রনাথ ঠাকুরের সমালোচনাসহ প্রকাশিত হত """" নামে। ১৮৮১ সালে সেই পত্রাবলি """" নামে গ্রন্থাকারে ছাপা হয়। এটিই ছিল রবীন্দ্রনাথের প্রথম গদ্যগ্রন্থ তথা প্রথম চলিত ভাষায় লেখা গ্রন্থ। অবশেষে ১৮৮০ সালে প্রায় দেড় বছর ইংল্যান্ডে কাটিয়ে কোনো ডিগ্রি না নিয়ে এবং ব্যারিস্টারি পড়া শুরু না করেই তিনি দেশে ফিরে আসেন।
```

``` {#b0b9 .graf .graf--pre .graf-after--pre name="b0b9"}
১৮৮৩ সালের ৯ ডিসেম্বর (২৪ অগ্রহায়ণ, ১২৯০ বঙ্গাব্দ) ঠাকুরবাড়ির অধস্তন কর্মচারী বেণীমাধব রায়চৌধুরীর কন্যা ভবতারিণীর সঙ্গে রবীন্দ্রনাথের বিবাহ সম্পন্ন হয়। বিবাহিত জীবনে ভবতারিণীর নামকরণ হয়েছিল মৃণালিনী দেবী (১৮৭৩–১৯০২ )। রবীন্দ্রনাথ ও মৃণালিনীর সন্তান ছিলেন পাঁচ জন: মাধুরীলতা (১৮৮৬–১৯১৮), রথীন্দ্রনাথ (১৮৮৮–১৯৬১), রেণুকা (১৮৯১–১৯০৩), মীরা (১৮৯৪–১৯৬৯) এবং শমীন্দ্রনাথ (১৮৯৬–১৯০৭)। এঁদের মধ্যে অতি অল্প বয়সেই রেণুকা ও শমীন্দ্রনাথের মৃত্যু ঘটে।
```

``` {#7d1c .graf .graf--pre .graf-after--pre name="7d1c"}
১৮৯১ সাল থেকে পিতার আদেশে নদিয়া (নদিয়ার উক্ত অংশটি অধুনা বাংলাদেশের কুষ্টিয়া জেলা), পাবনা ও রাজশাহী জেলা এবং উড়িষ্যার জমিদারিগুলির তদারকি শুরু করেন রবীন্দ্রনাথ। কুষ্টিয়ার শিলাইদহের কুঠিবাড়িতে রবীন্দ্রনাথ দীর্ঘ সময় অতিবাহিত করেছিলেন। জমিদার রবীন্দ্রনাথ শিলাইদহে ""পদ্মা"" নামে একটি বিলাসবহুল পারিবারিক বজরায় চড়ে প্রজাবর্গের কাছে খাজনা আদায় ও আশীর্বাদ প্রার্থনা করতে যেতেন। গ্রামবাসীরাও তাঁর সম্মানে ভোজসভার আয়োজন করত।
```

``` {#d9c1 .graf .graf--pre .graf-after--pre name="d9c1"}
১৮৯০ সালে রবীন্দ্রনাথের অপর বিখ্যাত কাব্যগ্রন্থ """" প্রকাশিত হয়। কুড়ি থেকে ত্রিশ বছর বয়সের মধ্যে তাঁর আরও কয়েকটি উল্লেখযোগ্য কাব্যগ্রন্থ ও গীতিসংকলন প্রকাশিত হয়েছিল। এগুলি হলো """", """", ""রবিচ্ছায়া"", """" ইত্যাদি। ১৮৯১ থেকে ১৮৯৫ সাল পর্যন্ত নিজের সম্পাদিত ""সাধনা"" পত্রিকায় রবীন্দ্রনাথের বেশ কিছু উৎকৃষ্ট রচনা প্রকাশিত হয়। তাঁর সাহিত্যজীবনের এই পর্যায়টি তাই ""সাধনা পর্যায়"" নামে পরিচিত। রবীন্দ্রনাথের ""গল্পগুচ্ছ"" গ্রন্থের প্রথম চুরাশিটি গল্পের অর্ধেকই এই পর্যায়ের রচনা। এই ছোটগল্পগুলিতে তিনি বাংলার গ্রামীণ জনজীবনের এক আবেগময় ও শ্লেষাত্মক চিত্র এঁকেছিলেন।
```

-   Preprocess the csv files with the following code using regular
    expressions (to get rid of punctuations). Remember we need to decode
    to utf-8 first, since we have unicode input files.

``` {#cdea .graf .graf--pre .graf-after--li name="cdea"}
from glob import globimport rewords = []for f in glob('bangla/wiki/*.csv'):    words += re.sub('[\r\n—?,;।!‘"’\.:\(\)\[\]…0-9]', ' ', \                    open(f, 'rb').read().decode('utf8').strip()) \                    .split(' ')words = list(filter(lambda x: not x in ['', '-'], words))print(len(words))# 13964346words[:25]#['রবীন্দ্রনাথ',# 'ঠাকুর',# 'রবীন্দ্রনাথ',# 'ঠাকুর',# '৭ই',# 'মে',# '১৮৬১',# '৭ই',# 'আগস্ট',# '১৯৪১',# '২৫',# 'বৈশাখ',# '১২৬৮',# '২২',# 'শ্রাবণ',# '১৩৪৮',# 'বঙ্গাব্দ',# 'ছিলেন',# 'অগ্রণী',# 'বাঙালি',# 'কবি',# 'ঔপন্যাসিক',# 'সংগীতস্রষ্টা',# 'নাট্যকার',# 'চিত্রকর']
```

-   Create indices for unique words in the dataset.

``` {#6f24 .graf .graf--pre .graf-after--li name="6f24"}
vocabulary_size = 25000
```

``` {#e5c2 .graf .graf--pre .graf-after--pre name="e5c2"}
def build_dataset(words):  count = [['UNK', -1]]  count.extend(collections.Counter(words).most_common( \               vocabulary_size - 1))  dictionary = dict()  for word, _ in count:    dictionary[word] = len(dictionary)  data = list()  unk_count = 0  for word in words:    if word in dictionary:      index = dictionary[word]    else:      index = 0  # dictionary['UNK']      unk_count = unk_count + 1    data.append(index)  count[0][1] = unk_count  reverse_dictionary = dict(zip(dictionary.values(), \                                dictionary.keys()))   return data, count, dictionary, reverse_dictionary
```

``` {#5f01 .graf .graf--pre .graf-after--pre name="5f01"}
data, count, dictionary, reverse_dictionary = build_dataset(words)print('Most common words (+UNK)', count[:5])# Most common words (+UNK) [['UNK', 1961151], ('এবং', 196916), ('ও', 180042), ('হয়', 160533), ('করে', 131206)]print('Sample data', data[:10])#Sample data [1733, 1868, 1733, 1868, 5769, 287, 6855, 5769, 400, 2570]del words  # Hint to reduce memory.
```

-   Generate batches to be trained with the word2vec model.
-   The target label should be at the center of the buffer each time.
    That is, given a focus word, our goal will be to learn the most
    probable context words.
-   The input and the target vector will depend on *num\_skips*and
    *skip\_window.*

``` {#a652 .graf .graf--pre .graf-after--li name="a652"}
data_index = 0
```

``` {#5bbf .graf .graf--pre .graf-after--pre name="5bbf"}
def generate_batch(batch_size, num_skips, skip_window):  global data_index  assert batch_size % num_skips == 0  assert num_skips <= 2 * skip_window  batch = np.ndarray(shape=(batch_size), dtype=np.int32)  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)  span = 2 * skip_window + 1 # [ skip_window target skip_window ]  buffer = collections.deque(maxlen=span)  for _ in range(span):    buffer.append(data[data_index])    data_index = (data_index + 1) % len(data)  for i in range(batch_size // num_skips):    target = skip_window  #     targets_to_avoid = [ skip_window ]    for j in range(num_skips):      while target in targets_to_avoid:        target = random.randint(0, span - 1)      targets_to_avoid.append(target)      batch[i * num_skips + j] = buffer[skip_window]      labels[i * num_skips + j, 0] = buffer[target]    buffer.append(data[data_index])    data_index = (data_index + 1) % len(data)  return batch, labels
```

``` {#5cd8 .graf .graf--pre .graf-after--pre name="5cd8"}
print('data:', [reverse_dictionary[di] for di in data[:8]])# data: ['রবীন্দ্রনাথ', 'ঠাকুর', 'রবীন্দ্রনাথ', 'ঠাকুর', '৭ই', 'মে', '১৮৬১', '৭ই']
```

``` {#5f02 .graf .graf--pre .graf-after--pre name="5f02"}
for num_skips, skip_window in [(2, 1), (4, 2)]:    data_index = 0    batch, labels = generate_batch(batch_size=8, \                      num_skips=num_skips, skip_window=skip_window)    print('\nwith num_skips = %d and skip_window = %d:' %           (num_skips, skip_window))    print('    batch:', [reverse_dictionary[bi] for bi in batch])    print('    labels:', [reverse_dictionary[li] for li in \                                              labels.reshape(8)])
```

``` {#a53e .graf .graf--pre .graf-after--pre name="a53e"}
# data: ['রবীন্দ্রনাথ', 'ঠাকুর', 'রবীন্দ্রনাথ', 'ঠাকুর', '৭ই', 'মে',   #        '১৮৬১', '৭ই']# with num_skips = 2 and skip_window = 1:# batch: ['ঠাকুর', 'ঠাকুর', 'রবীন্দ্রনাথ', 'রবীন্দ্রনাথ', 'ঠাকুর', 'ঠাকুর',  #        '৭ই', '৭ই']# labels: ['রবীন্দ্রনাথ', 'রবীন্দ্রনাথ', 'ঠাকুর', 'ঠাকুর', '৭ই', 'রবীন্দ্রনাথ', #'ঠাকুর', 'মে']# with num_skips = 4 and skip_window = 2:# batch: ['রবীন্দ্রনাথ', 'রবীন্দ্রনাথ', 'রবীন্দ্রনাথ', 'রবীন্দ্রনাথ', 'ঠাকুর', #    # 'ঠাকুর', 'ঠাকুর', 'ঠাকুর']# labels: ['রবীন্দ্রনাথ', '৭ই', 'ঠাকুর', 'ঠাকুর', 'মে', 'ঠাকুর', 'রবীন্দ্রনাথ', # '৭ই']
```

-   Pick a random validation set to sample nearest neighbors.
-   Limit the validation samples to the words that have a low numeric
    ID, which by construction are also the most frequent.
-   Look up embeddings for inputs and compute the softmax loss, using a
    sample of the negative labels each time (this is known as negative
    sampling, which is used to make the computation efficient, since the
    number of labels are often too high).
-   The optimizer will optimize the softmax\_weights and the
    embeddings.\
    This is because the embeddings are defined as a variable quantity
    and the optimizer’s \`minimize\` method will by default modify all
    variable quantities that contribute to the tensor it is passed.
-   Compute the similarity between minibatch examples and all
    embeddings.

``` {#9f84 .graf .graf--pre .graf-after--li name="9f84"}
batch_size = 128embedding_size = 128 # Dimension of the embedding vector.skip_window = 1 # How many words to consider left and right.num_skips = 2 # #times to reuse an input to generate a label. valid_size = 16 # Random set of words to evaluate similarity on.valid_window = 100 # Only pick dev samples in the head of the                      # distribution.valid_examples = np.array(random.sample(range(valid_window),                                         valid_size))num_sampled = 64 # Number of negative examples to sample.
```

``` {#6f18 .graf .graf--pre .graf-after--pre name="6f18"}
graph = tf.Graph()
```

``` {#03e2 .graf .graf--pre .graf-after--pre name="03e2"}
with graph.as_default(), tf.device('/cpu:0'):
```

``` {#8c46 .graf .graf--pre .graf-after--pre name="8c46"}
  # Input data.  train_dataset = tf.placeholder(tf.int32, shape=[batch_size])  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)    # Variables.  embeddings = tf.Variable( \    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))  softmax_weights = tf.Variable( \    tf.truncated_normal([vocabulary_size, embedding_size], \                         stddev=1.0 / math.sqrt(embedding_size)))  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))    # Model.  embed = tf.nn.embedding_lookup(embeddings, train_dataset)    loss = tf.reduce_mean( \    tf.nn.sampled_softmax_loss(weights=softmax_weights, \                            biases=softmax_biases, inputs=embed, \                                         labels=train_labels, \          num_sampled=num_sampled, num_classes=vocabulary_size))
```

``` {#e883 .graf .graf--pre .graf-after--pre name="e883"}
# Optimizer.# See docs on `tf.train.Optimizer.minimize()` for more details.optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)  # use the cosine distance:  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, \                               keepdims=True))  normalized_embeddings = embeddings / norm  valid_embeddings = tf.nn.embedding_lookup( \                     normalized_embeddings, valid_dataset)  similarity = tf.matmul(valid_embeddings,   \                         tf.transpose(normalized_embeddings))
```

-   Train the word2vec model with the batches constructed, for 100k
    steps.

``` {#8f86 .graf .graf--pre .graf-after--li name="8f86"}
num_steps = 100001
```

``` {#367d .graf .graf--pre .graf-after--pre name="367d"}
with tf.Session(graph=graph) as session:  tf.global_variables_initializer().run()  print('Initialized')  average_loss = 0  for step in range(num_steps):    batch_data, batch_labels = generate_batch(      batch_size, num_skips, skip_window)    feed_dict = {train_dataset : batch_data, \                 train_labels : batch_labels}    _, l = session.run([optimizer, loss], feed_dict=feed_dict)    average_loss += l    if step % 2000 == 0:      if step > 0:        average_loss = average_loss / 2000      # The average loss is an estimate of the loss over the last       # 2000 batches.      print('Average loss at step %d: %f' % (step, average_loss))      average_loss = 0      # note that this is expensive (~20% slowdown if computed every       # 500 steps)    if step % 10000 == 0:      sim = similarity.eval()      for i in range(valid_size):        valid_word = reverse_dictionary[valid_examples[i]]        top_k = 8 # number of nearest neighbors        nearest = (-sim[i, :]).argsort()[1:top_k+1]        log = 'Nearest to %s:' % valid_word        for k in range(top_k):          close_word = reverse_dictionary[nearest[k]]          log = '%s %s,' % (log, close_word)        print(log)  final_embeddings = normalized_embeddings.eval()
```

-   The following shows how the loss function decreases with the
    increase in training steps.
-   During the training process, the words that become semantically near
    come closer in the embedding space.

![](https://cdn-images-1.medium.com/max/800/1*sgn7umWkdT787e73ZuTkGA.png)

Image by Author

-   Use t-SNE plot to map the following words from 128-dimensional
    embedding space to 2 dimensional manifold and visualize.

``` {#6579 .graf .graf--pre .graf-after--li name="6579"}
words = ['রাজা', 'রাণী', 'ভারত','বাংলাদেশ','দিল্লী','কলকাতা','ঢাকা',         'পুরুষ','নারী','দুঃখ','লেখক','কবি','কবিতা','দেশ',         'বিদেশ','লাভ','মানুষ', 'এবং', 'ও', 'গান', 'সঙ্গীত', 'বাংলা',          'ইংরেজি', 'ভাষা', 'কাজ', 'অনেক', 'জেলার', 'বাংলাদেশের',          'এক', 'দুই', 'তিন', 'চার', 'পাঁচ', 'দশ', '১', '৫', '২০',          'নবম', 'ভাষার', '১২', 'হিসাবে', 'যদি', 'পান', 'শহরের', 'দল',          'যদিও', 'বলেন', 'রান', 'করেছে', 'করে', 'এই', 'করেন', 'তিনি',          'একটি', 'থেকে', 'করা', 'সালে', 'এর', 'যেমন', 'সব',  'তার',          'খেলা',  'অংশ', 'উপর', 'পরে', 'ফলে',  'ভূমিকা', 'গঠন',           'তা', 'দেন', 'জীবন', 'যেখানে', 'খান', 'এতে',  'ঘটে', 'আগে',          'ধরনের', 'নেন', 'করতেন', 'তাকে', 'আর', 'যার', 'দেখা',          'বছরের', 'উপজেলা', 'থাকেন', 'রাজনৈতিক', 'মূলত', 'এমন',          'কিলোমিটার', 'পরিচালনা', '২০১১', 'তারা', 'তিনি', 'যিনি', 'আমি',           'তুমি', 'আপনি', 'লেখিকা', 'সুখ', 'বেদনা', 'মাস', 'নীল', 'লাল',          'সবুজ', 'সাদা', 'আছে', 'নেই', 'ছুটি', 'ঠাকুর',         'দান', 'মণি', 'করুণা', 'মাইল', 'হিন্দু', 'মুসলমান','কথা', 'বলা',              'সেখানে', 'তখন', 'বাইরে', 'ভিতরে', 'ভগবান' ]indices = []for word in words:    #print(word, dictionary[word])    indices.append(dictionary[word])two_d_embeddings = tsne.fit_transform(final_embeddings[indices, :])plot(two_d_embeddings, words)
```

-   The following figure shows how the words similar in meaning are
    mapped to embedding vectors that are close to each other.
-   Also, note that arithmetic property of the word embeddings: e.g.,
    the words ‘রাজা’ and ‘রাণী’ are approximately along the same
    distance and direction as the words ‘লেখক’ and ‘লেখিকা’, reflecting
    the fact that the nature of the semantic relatedness in terms of
    gender is same.

![](https://cdn-images-1.medium.com/max/800/1*snzp25Pd4VZd8TUpWzrSoA.png)

Image by Author

-   The following animation shows how the embedding is learnt to
    preserve the semantic similarity in the 2D-manifold more and more as
    training proceeds.

![](https://cdn-images-1.medium.com/max/800/1*hWY1PQlG7q4xhu3jHUvJVQ.gif)

Image by Author

### Generating song-like texts with LSTM from Tagore’s Bangla songs {#af26 .graf .graf--h3 .graf-after--figure name="af26"}

**Text generation with Character LSTM**

-   Let’s import the required libraries first.

``` {#a352 .graf .graf--pre .graf-after--li name="a352"}
from tensorflow.keras.callbacks import LambdaCallbackfrom tensorflow.keras.models import Sequentialfrom tensorflow.keras.layers import Densefrom tensorflow.keras.layers import LSTMfrom tensorflow.keras.optimizers import RMSprop, Adamimport io, re
```

-   Read the input file, containing few selected songs of Tagore in
    Bangla.

``` {#27e5 .graf .graf--pre .graf-after--li name="27e5"}
raw_text = open('rabindrasangeet.txt','rb').read().decode('utf8')print(raw_text[0:1000])
```

``` {#e26a .graf .graf--pre .graf-after--pre name="e26a"}
পূজা
```

``` {#ba1b .graf .graf--pre .graf-after--pre name="ba1b"}
অগ্নিবীণা বাজাও তুমিঅগ্নিবীণা বাজাও তুমি কেমন ক’রে !আকাশ কাঁপে তারার আলোর গানের ঘোরে ।।তেমনি ক’রে আপন হাতে ছুঁলে আমার বেদনাতে,নূতন সৃষ্টি জাগল বুঝি জীবন-‘পরে ।।বাজে ব’লেই বাজাও তুমি সেই গরবে,ওগো প্রভু, আমার প্রাণে সকল সবে ।বিষম তোমার বহ্নিঘাতে বারে বারে আমার রাতেজ্বালিয়ে দিলে নূতন তারা ব্যথায় ভ’রে ।। অচেনাকে ভয় কীঅচেনাকে ভয় কী আমার ওরে?অচেনাকেই চিনে চিনে উঠবে জীবন ভরে ।।জানি জানি আমার চেনা কোনো কালেই ফুরাবে না,চিহ্নহারা পথে আমায় টানবে অচিন ডোরে ।।ছিল আমার মা অচেনা, নিল আমায় কোলে ।সকল প্রেমই অচেনা গো, তাই তো হৃদয় দোলে ।অচেনা এই ভুবন-মাঝে কত সুরেই হৃদয় বাজে-অচেনা এই জীবন আমার, বেড়াই তারি ঘোরে ।।
```

``` {#c323 .graf .graf--pre .graf-after--pre name="c323"}
অন্তর মমঅন্তর মম বিকশিত করো অন্তরতর হে-নির্মল করো, উজ্জ্বল করো, সুন্দর করো হে ।।জাগ্রত করো, উদ্যত করো, নির্ভয় করো হে ।।মঙ্গল করো, নিরলস নিঃসংশয় করো হে ।।যুক্ত করো হে সবার সঙ্গে, মুক্ত করো হে বন্ধ ।সঞ্চার করো সকল কর্মে শান্ত তোমার ছন্দ ।চরণপদ্মে মম চিত নিস্পন্দিত করো হে ।নন্দিত করো, নন্দিত করো, নন্দিত করো হে ।। অন্তরে জাগিছ অন্তর্যামীঅন্তরে জাগিছ অন্তর্যামী ।
```

-   Pre-process the text and create word indices to be used as the input
    in the model.

``` {#06be .graf .graf--pre .graf-after--li name="06be"}
processed_text = raw_text.lower()word_index = [i+1 for i, c in enumerate(processed_text) if c == ' ']print('corpus length:', len(processed_text))# corpus length: 207117chars = sorted(list(set(processed_text)))print('total chars:', len(chars))# total chars: 89char_indices = dict((c, i) for i, c in enumerate(chars))indices_char = dict((i, c) for i, c in enumerate(chars))
```

-   Cut the text in semi-redundant sequences of maxlen characters.

``` {#8560 .graf .graf--pre .graf-after--li name="8560"}
def is_conjunction(c):  h = ord(c) # print(hex(ord(c)))  return (h >= 0x980 and h <= 0x983) or          (h >= 0x9bc and h <= 0x9e5) or (h >= 0x9f2)
```

``` {#6ccb .graf .graf--pre .graf-after--pre name="6ccb"}
maxlen = 40step = 2sentences = []next_chars = []i = 0while i < len(processed_text) - maxlen:  if is_conjunction(processed_text[i]):    i += 1    continue  sentences.append(processed_text[i: i + maxlen])  next_chars.append(processed_text[i + maxlen])  i += step  print('nb sequences:', len(sentences))
```

``` {#6f6e .graf .graf--pre .graf-after--pre name="6f6e"}
# nb sequences: 89334
```

-   Create one-hot-encodings.

``` {#3baf .graf .graf--pre .graf-after--li name="3baf"}
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)y = np.zeros((len(sentences), len(chars)), dtype=np.bool)for i, sentence in enumerate(sentences):  for t, char in enumerate(sentence):    x[i, t, char_indices[char]] = 1    y[i, char_indices[next_chars[i]]] = 1
```

-   Build a model, a single LSTM.

``` {#ac59 .graf .graf--pre .graf-after--li name="ac59"}
model = Sequential()model.add(LSTM(256, input_shape=(maxlen, len(chars))))model.add(Dense(128, activation='relu'))model.add(Dense(len(chars), activation='softmax'))optimizer = Adam(lr=0.01) #RMSprop(lr=0.01)model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

-   The following figure how the model architecture looks like:

![](https://cdn-images-1.medium.com/max/800/1*OxLFyS9dPuTEybKrlRkx3Q.png)

Image by Author

-   Print the model summary.

``` {#65c9 .graf .graf--pre .graf-after--li name="65c9"}
model.summary()
```

``` {#fa29 .graf .graf--pre .graf-after--pre name="fa29"}
Model: "sequential" _________________________________________________________________ Layer (type)                 Output Shape              Param #    ================================================================= lstm (LSTM)                  (None, 256)               354304     _________________________________________________________________ dense (Dense)                (None, 128)               32896      _________________________________________________________________ dense_1 (Dense)              (None, 89)                11481      ================================================================= Total params: 398,681 Trainable params: 398,681 Non-trainable params: 0 _________________________________________________________________
```

-   Use the following helper function to sample an index from a
    probability array.

``` {#a677 .graf .graf--pre .graf-after--li name="a677"}
def sample(preds, temperature=1.0):  preds = np.asarray(preds).astype('float64')  preds = np.log(preds) / temperature  exp_preds = np.exp(preds)  preds = exp_preds / np.sum(exp_preds)  probas = np.random.multinomial(1, preds, 1)  return np.argmax(probas)
```

-   Fit the model and register a callback to print the text generated by
    the model at the end of each epoch.

``` {#5a3c .graf .graf--pre .graf-after--li name="5a3c"}
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)model.fit(x, y, batch_size=128, epochs=60,           callbacks=[print_callback])
```

-   The following animation shows how the model generates song-like
    texts with given seed texts, for different values of the temperature
    parameter.

![](https://cdn-images-1.medium.com/max/800/1*Gd6z7o24c-OnupPpVkcsAQ.gif)

Image by Author

**Text Generation with Word LSTM**

to be continued…

By [Sandipan Dey](https://medium.com/@sandipan-dey) on [January 2,
2021](https://medium.com/p/770433f6a06d).

[Canonical
link](https://medium.com/@sandipan-dey/nlp-with-bangla-semantic-similarity-with-word2vec-and-generating-bangla-text-with-lstm-770433f6a06d)

Exported from [Medium](https://medium.com) on January 8, 2021.
