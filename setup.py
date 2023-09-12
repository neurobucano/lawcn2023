from setuptools import setup, find_packages

setup(
    name='lawcn2023',
    version='0.1.0',
    description='Descrição do seu projeto',
    author='Nivaldo A P de Vasconcelos',
    author_email='nivaldo.vasconcelos@ufpe.br',
    packages=find_packages(),  # Isso procura automaticamente por todos os pacotes do seu projeto
    install_requires=[
        'pandas', 'numpy' # Certifique-se de listar todas as dependências necessárias aqui
    ],
)


