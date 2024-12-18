# LigandDesigner
LigandDesigner: A machine learning-driven system for ligand design integrating pharmacophore modeling, generative models (GAN, LSTM), and molecular dynamics simulations. Includes versions for HomeLab (local) and HPC (high-performance computing).

# LigandDesigner

LigandDesigner to zestaw narzędzi do **projektowania ligandów** opartych na technologiach **uczenia maszynowego** oraz **modelowania molekularnego**. System zawiera dwie wersje:

- **LigandDesigner-HomeLab** – zoptymalizowany do pracy w środowiskach o ograniczonych zasobach obliczeniowych.
- **LigandDesigner-HPC** – zoptymalizowany do pracy w środowiskach klastrowych i superkomputerach z obsługą GPU i obliczeń rozproszonych.

---

## 📋 Wymagania systemowe

### HomeLab:
- Python 3.8+
- CPU (opcjonalnie GPU, np. NVIDIA RTX lub GTX)
- Zainstalowane CUDA (dla GPU)

### HPC:
- Python 3.8+
- Klaster obliczeniowy lub superkomputer z:
  - NVIDIA GPU (np. V100/A100)
  - CUDA Toolkit 11.8+
  - OpenMPI do obliczeń rozproszonych

---

## 🚀 Instalacja

1. **Klonowanie repozytorium**:
   ```bash
   git clone https://github.com/username/LigandDesigner.git
   cd LigandDesigner

2. **Utworzenie środowiska wirtualnego:**
  python -m venv venv
  source venv/bin/activate   # Linux/Mac
  venv\Scripts\activate      # Windows

3. **Instalacja wymaganych bibliotek:**
   pip install -r requirements.txt

4. **Weryfikacja instalacji:**
   python -c "import torch; print(torch.__version__)"

## ⚙️ Uruchamianie system

HomeLab:
Uruchomienie projektu w lokalnym środowisku:

python ligand_designer_homelab.py --config=config_homelab.yaml

HPC:
Uruchamianie na klastrze z obliczeniami rozproszonymi:

mpirun -np 8 python ligand_designer_hpc.py --config=config_hpc.yaml

## 📂 Struktura projektu
LigandDesigner/
│-- ligand_designer_homelab.py    # Kod dla HomeLab
│-- ligand_designer_hpc.py        # Kod dla HPC
│-- configs/
│   ├── config_homelab.yaml       # Konfiguracja dla HomeLab
│   └── config_hpc.yaml           # Konfiguracja dla HPC
│-- requirements.txt              # Wymagane biblioteki
└-- README.md                     # Dokumentacja

## Licencja
Ten projekt jest licencjonowany na zasadach MIT License. Szczegóły znajdziesz w pliku LICENSE.

## Powiązane publikacje
1. https://doi.org/10.5281/zenodo.14512310
2. https://biorxiv.org/cgi/content/short/2024.12.01.626243v1
3. https://www.qeios.com/read/1CLGH0
4. https://doi.org/10.6084/m9.figshare.28038599.v1

## ✉️ Kontakt

Jeśli masz pytania lub sugestie dotyczące LigandDesigner, skontaktuj się:

    Imię i nazwisko: Sławomir Wiśniewski
    E-mail: sa.wisniewski@sci4biz.edu.pl





