from django import forms


class SymptomForm(forms.Form):
    SYSTEMIC_ILLNESS_CHOICES = [
        ('None', 'None'),
        ('Fever', 'Fever'),
        ('Muscle Aches and Pain', 'Muscle Aches and Pain'),
        ('Swollen Lymph Nodes', 'Swollen Lymph Nodes'),
    ]

    systemic_illness = forms.ChoiceField(
        choices=SYSTEMIC_ILLNESS_CHOICES, 
        required=True, 
        label='Systemic Illness'
    )
    sore_throat = forms.BooleanField(required=False, initial=False)
    rectal_pain = forms.BooleanField(required=False, initial=False)
    penile_oedema = forms.BooleanField(required=False, initial=False)
    oral_lesions = forms.BooleanField(required=False, initial=False)
    solitary_lesion = forms.BooleanField(required=False, initial=False)
    swollen_tonsils = forms.BooleanField(required=False, initial=False)
    hiv_infection = forms.BooleanField(required=False, initial=False)
    sexually_transmitted_infection = forms.BooleanField(required=False, initial=False)


class PossumPredictionForm(forms.Form):
    SITE_CHOICES = [(i, str(i)) for i in range(1, 8)]  # Options 1 through 7
    POP_CHOICES = [('Vic', 'Vic'), ('other', 'Other')]  # Vic/Other choices
    SEX_CHOICES = [('m', 'Male'), ('f', 'Female')]  # Male/Female choices

    site = forms.ChoiceField(choices=SITE_CHOICES, label="Site")
    Pop_other = forms.ChoiceField(choices=POP_CHOICES, label="Pop")
    sex_m = forms.ChoiceField(choices=SEX_CHOICES, label="Sex")
    age = forms.FloatField(label="Age")
    skullw = forms.FloatField(label="Skull Width")
    totlngth = forms.FloatField(label="Total Length")
    taill = forms.FloatField(label="Tail Length")
    footlgth = forms.FloatField(label="Foot Length")
    earconch = forms.FloatField(label="Ear Conch")
    eye = forms.FloatField(label="Eye Width")
    chest = forms.FloatField(label="Chest Width")
    belly = forms.FloatField(label="Belly Width")