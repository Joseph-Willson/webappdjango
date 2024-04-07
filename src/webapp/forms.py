from django import forms

class MyForm(forms.Form):
    my_field = forms.CharField(max_length=1000, widget=forms.TextInput(attrs={'placeholder': 'Entrez votre texte ici', 'class': 'custom-field'}))