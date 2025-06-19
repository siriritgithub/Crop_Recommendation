# fertilizer_recommender.py

fertilizer_dict = {
    'rice': {
        'N': 90,
        'P': 40,
        'K': 40,
        'recommendation': 'Use Urea for N, DAP for P, and MOP for K.',
        'note': 'Ensure flooded conditions for rice during early growth stages.',
        'pesticide': 'Use Carbendazim for fungal infections; apply insecticide for stem borer.'
    },
    'wheat': {
        'N': 100,
        'P': 50,
        'K': 50,
        'recommendation': 'Apply Urea and Single Super Phosphate. MOP for K.',
        'note': 'Top-dress Urea in two splits.',
        'pesticide': 'Use Imidacloprid for aphids and foliar spray for rust control.'
    },
    'maize': {
        'N': 120,
        'P': 60,
        'K': 40,
        'recommendation': 'Nitrogen from Urea; P from DAP; K from MOP.',
        'note': 'Top dressing at knee-high stage.',
        'pesticide': 'Chlorpyrifos for cutworms; spray fungicides for blight.'
    },
    'cotton': {
        'N': 75,
        'P': 40,
        'K': 40,
        'recommendation': 'Urea for N; complex fertilizer if available.',
        'note': 'Ensure pest control regularly.',
        'pesticide': 'Neem oil or Spinosad for bollworms and whiteflies.'
    },
    'sugarcane': {
        'N': 150,
        'P': 50,
        'K': 75,
        'recommendation': 'Heavy feeders. Use split doses.',
        'note': 'Add FYM during land prep.',
        'pesticide': 'Use systemic pesticides for borers and white grubs.'
    }
}


def get_fertilizer_recommendation(crop_name):
    crop_name = crop_name.lower()
    return fertilizer_dict.get(crop_name, {
        'N': 'Unknown',
        'P': 'Unknown',
        'K': 'Unknown',
        'recommendation': 'No recommendation available for this crop.',
        'note': ''
    })
