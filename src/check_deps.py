"""Check all required dependencies."""

required_packages = [
    ('pandas', 'pd'),
    ('numpy', 'np'), 
    ('matplotlib.pyplot', 'plt'),
    ('seaborn', 'sns'),
    ('plotly.express', 'px'),
    ('plotly.graph_objects', 'go'),
    ('sklearn.preprocessing', 'StandardScaler'),
    ('scipy.spatial.distance', 'pdist'),
    ('os', None)
]

missing_packages = []

for package, alias in required_packages:
    try:
        if alias:
            exec(f"import {package} as {alias}")
        else:
            exec(f"import {package}")
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package}")
        missing_packages.append(package.split('.')[0])

if missing_packages:
    print(f"\n📦 Install missing packages:")
    print(f"pip install {' '.join(set(missing_packages))}")
else:
    print(f"\n🎉 All dependencies available!")
