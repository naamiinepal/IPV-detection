# Detecting Online Intimate Partner Violence using AI
### In collaboration with ChildSafeNet.
### Grant: Sexual Violence Research Initiative (SVRI).

## Introduction
- IPV stands for Intimate Partner Violence.
- It is Mostly manifested in physical, sexual and emotional forms.
- The project aims to study the prevalence of violent and abusive behaviour in an intimate relationship (on ChildSafeNet's end) and also create an AI tool to help identify potential IPV containing texts in digital platform (on our end).
- There is an increasing prevalence in dating relationships in teenagers and youths, especially in online contexts.
- No research has been conducted for IPV in digital platform.

## Study Schemes
### Fine-grained Scheme
- TASK: Identify *aspect terms* and assign *aspect category + polarity*.
- Studies the presence of IPV content in a phrase level.
- Specific. May lose context in the mix.

### Coarse-grained Scheme
- TASK: Sentence level IPV classification.
- Studies the presence of IPV content on a sentence level.
- Context somewhat preserved. But may not be specific when a single sentence carries two or more sentiment bearing component. 

## Downstream Tasks:
The project consists of two main tasks, **Aspect Term Extraction** and **IPV Polarity Classification**.

#### **Aspect Term Extraction**
- A sequence labelling problem, in which each individual tokens are mapped to their corresponding aspect categories.
- Note that the aspect categories may be one of: 
    - Profanity
    - Character assassination
    - Emotional abuse
    - Physical threat
    - Rape threat
    - General threat
    - Violence based on ethnicity 
    - Violence based on religion
    - Sexism
    - Others

#### **IPV Polarity Classification**
- Here, we map the text into one of three categories: {IPV, non-IPV, Unknown}.
- There are three modes under which we can train the model.
    1. Text --> Polarity
    2. Text + aspect_term --> Polarity
    3. Text + aspect_term + aspect_category --> Polarity

## Annotation:
- Used a web-based annotation tool called WebAnno.
- Methodology:
    - Created batches of 10 sentences from the curated data.
    - Annotations created by the two interns at ChildSafeNet and exported into a TSV file.
    - A subset (almost 15%) of the total data annotated by both to check the inter-annotator agreement.
    - A parser constructed to convert the data into a feasible input format for the aspect extraction model and the IPV classifier.  

### Inter-annotator agreement:
- Measured using weighted F1 Score
- <img src="https://render.githubusercontent.com/render/math?math=F-measure = \frac{(\beta^2 + 1) X Precision X Recall}{(\beta^2 X precision) \+ Recall}">

#### Precision w.r.t Tag 'T':
<img src="https://render.githubusercontent.com/render/math?math=Pr_{T}(A_{1}, A_{2}) = \frac{Number of tokens that A_{1} marked 'T' and A_{2} marked 'T'}{Number of tokens that A_{1} marked 'T'}">

#### Recall w.r.t Tag 'T':
<img src="https://render.githubusercontent.com/render/math?math=Rec_{T}(A_{1}, A_{2}) = \frac{Number of tokens that A_{1} marked 'T' and A_{2} marked 'T'}{Number of tokens that A_{2} marked 'T'}">

## Web Crawling:
- Curated search terms (queries) for twitter scraper.
- Created Term Frequency in IPV data.
- Excluded stopwords.
- Selected highest freq words.
- Handpicked some from the low freq words too.
- Used variations of words.
    - अलच्छिनी OR अलक्षिनि
    - बलात्कारी OR बलत्कार OR बलात्कार

- Total 36 search terms.

### Data curated frequency
![Data curated frequency](images/markdown/data_collection_stat.png "Data curated frequency")


