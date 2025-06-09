# TTT-RNN Individual Plots Summary

This directory contains individual, high-quality plots for the TTT-RNN phoneme decoding analysis. Each plot is saved separately at 300 DPI for publication-quality figures.

## 📊 Plot Categories and Files

### Performance Metrics (`performance/`)
- **overall_metrics.png** - Bar chart showing CER, PER, and Frame Accuracy

### Sequence Length Analysis (`sequences/`)
- **cer_by_length.png** - Character Error Rate by sequence length bins
- **per_by_length.png** - Phoneme Error Rate by sequence length bins  
- **count_by_length.png** - Sample distribution across sequence lengths

### Phoneme Category Analysis (`phonemes/`)
- **accuracy_by_category.png** - Horizontal bar chart of accuracy by phoneme type
- **count_distribution.png** - Pie chart showing distribution of phoneme samples

### Confusion Analysis (`confusion/`)
- **confusion_pairs.png** - Most frequent phoneme confusion pairs with error types
- **confusion_matrix.png** - Full confusion matrix heatmap (simulated)

### TTT Adaptation Analysis (`ttt_adaptation/`)
- **loss_reduction.png** - TTT reconstruction loss before vs after adaptation
- **context_loss.png** - TTT loss comparison between phoneme boundaries and steady-state

### Temporal Dynamics (`temporal/`)
- **transition_latency.png** - Distribution of phoneme transition latencies
- **confidence_by_type.png** - Prediction confidence for vowels, consonants, transitions
- **boundary_detection.png** - Sentence boundary detection accuracy (pie chart)

### Error Distribution (`errors/`)
- **error_distribution.png** - Distribution of sentences by error rate categories

### Feature Analysis (`features/`)
- **neural_importance.png** - Relative importance of neural features (TX, spike power)
- **spatial_contributions.png** - Brain area contributions (Area 6v vs Area 44)

## 📈 Key Results Highlighted

- **Overall CER**: 38.0%
- **TTT Loss Reduction**: 37.8% (0.82 → 0.51)
- **Best Phoneme Category**: Long vowels (72.4% accuracy)
- **Perfect Sentences**: 142 out of 2,059 (6.9%)

## 🎨 Plot Features

- High resolution (300 DPI)
- Publication-ready formatting
- Clear value labels on all charts
- Consistent color schemes
- Professional styling with seaborn
- Individual files for easy insertion into papers/presentations

## 📝 Usage

Each plot can be used independently in:
- Research papers
- Conference presentations  
- Thesis documents
- Grant applications
- Technical reports

All plots are saved in PNG format with transparent backgrounds where appropriate. 