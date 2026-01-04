"""
data_ingestion.py - Download de dados de ações usando Yahoo Finance

Este módulo baixa dados históricos de ações e salva em formato Parquet.
Inclui tratamento robusto de erros e validações.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import time


def download_stock_data(ticker: str, period: str = "2y", output_dir: str = "./data") -> Path:
    """
    Baixa dados históricos de uma ação com tratamento robusto de erros.
    
    Args:
        ticker: Código da ação (ex: "PETR4.SA", "SPY")
        period: Período de dados (ex: "1y", "2y", "5y")
        output_dir: Diretório para salvar os dados
    
    Returns:
        Path: Caminho do arquivo salvo
        
    Raises:
        ValueError: Se os dados baixados estiverem vazios ou inválidos
        RuntimeError: Se houver erro na comunicação com Yahoo Finance
    """
    print(f"📥 Baixando dados de {ticker} (período: {period})...")
    
    max_retries = 3
    retry_delay = 2  # segundos
    
    for attempt in range(1, max_retries + 1):
        try:
            # Baixar dados do Yahoo Finance
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            # Validar se os dados foram baixados
            if df is None or df.empty:
                if attempt < max_retries:
                    print(f"⚠️  Tentativa {attempt}/{max_retries} falhou. Dados vazios. Tentando novamente em {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise ValueError(
                        f"Não foi possível baixar dados para {ticker}. "
                        f"Verifique se o ticker está correto e se você tem conexão com a internet. "
                        f"Tickers brasileiros devem terminar com .SA (ex: PETR4.SA)"
                    )
            
            # Validar se tem dados suficientes
            if len(df) < 100:
                raise ValueError(
                    f"Dados insuficientes para {ticker}. "
                    f"Apenas {len(df)} registros foram baixados. "
                    f"Mínimo recomendado: 100 registros."
                )
            
            # Renomear colunas para minúsculas
            df.columns = [col.lower() for col in df.columns]
            
            # Validar colunas necessárias
            required_columns = ['close', 'high', 'low', 'open', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(
                    f"Colunas obrigatórias ausentes: {missing_columns}. "
                    f"Colunas disponíveis: {list(df.columns)}"
                )
            
            # Criar diretório se não existir
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Salvar em formato Parquet
            file_path = output_path / f"{ticker.lower().replace('.', '_')}.parquet"
            df.to_parquet(file_path)
            
            print(f"✅ Dados salvos em: {file_path}")
            print(f"   Total de registros: {len(df)}")
            print(f"   Período: {df.index[0].strftime('%Y-%m-%d')} até {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Colunas: {list(df.columns)}")
            
            return file_path
            
        except Exception as e:
            if attempt < max_retries:
                print(f"⚠️  Tentativa {attempt}/{max_retries} falhou: {str(e)}")
                print(f"   Tentando novamente em {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                # Última tentativa falhou
                error_msg = (
                    f"Erro ao baixar dados de {ticker} após {max_retries} tentativas.\n"
                    f"Erro: {str(e)}\n\n"
                    f"Possíveis soluções:\n"
                    f"1. Verifique sua conexão com a internet\n"
                    f"2. Verifique se o ticker está correto\n"
                    f"3. Para ações brasileiras, use o formato: PETR4.SA, VALE3.SA, etc.\n"
                    f"4. Tente um período menor (ex: '1y' ao invés de '2y')\n"
                    f"5. O Yahoo Finance pode estar temporariamente indisponível\n"
                )
                raise RuntimeError(error_msg) from e
    
    # Nunca deve chegar aqui, mas por segurança
    raise RuntimeError(f"Falha inesperada ao baixar dados de {ticker}")


if __name__ == "__main__":
    # Exemplo de uso
    import sys
    
    try:
        ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
        period = sys.argv[2] if len(sys.argv) > 2 else "2y"
        
        print(f"Testando download de {ticker}...")
        file_path = download_stock_data(ticker, period)
        print(f"\n✅ Teste concluído com sucesso!")
        print(f"Arquivo salvo em: {file_path}")
        
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
        sys.exit(1)
