"""WindowsのPETSc環境を、MPI DLL設定を含むJupyterカーネルとして登録する。"""
import argparse
import json
from pathlib import Path
import tempfile


def kernel_spec(environment):
    """指定したWindows Conda環境用のJupyter kernelspecを作る。

    Parameters
    ----------
    environment : str or Path
        ipykernel・PETSc・MPIを導入したConda環境のフォルダ。

    Returns
    -------
    dict
        Pythonの直接起動、MPIランタイムのPATH・I_MPI_ROOT、FiPyのPETSc選択を含む設定。

    Raises
    ------
    FileNotFoundError
        Python実行ファイルまたはLibrary/binがない場合。

    Notes
    -----
    -Eで外側のPYTHONPATHを無視し、-sでユーザーsite-packagesの混入を防ぐ。
    PATHの${PATH}はJupyterが起動元の値へ展開する。DLL設定はこのカーネルの子プロセスだけに適用する。
    システム全体のPATHは変更しない。Windows用の環境定義を前提とする。

    Examples
    --------
    プロジェクトフォルダの親に環境がある場合::

        spec = kernel_spec('../env_pyaxistreamer_petsc')
        assert spec['env']['FIPY_SOLVERS'] == 'petsc'
    """
    environment = Path(environment).resolve()
    python = environment/'python.exe'
    library = environment/'Library'
    if not python.is_file() or not (library/'bin').is_dir():
        raise FileNotFoundError(f'Windows Conda environment not found: {environment}')
    return dict(argv=[str(python), '-E', '-s', '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='PyAxiStreamer (PETSc)', language='python',
        env={'PATH': ';'.join(map(str, (environment, library/'bin', environment/'Scripts')))+';${PATH}',
             'I_MPI_ROOT': str(library), 'CONDA_PREFIX': str(environment), 'FIPY_SOLVERS': 'petsc'})


def main():
    """PETScのJupyterカーネルを登録するか、確認用の設定だけを保存する。

    Returns
    -------
    int
        成功時は0。作成・登録先を標準出力へ表示する。

    Raises
    ------
    FileNotFoundError
        指定したWindows Conda環境がない場合。
    ImportError
        登録時にjupyter_clientが利用できない場合。ipykernelの導入で利用可能になる。

    Notes
    -----
    登録名はpyaxistreamer-petsc。同名の登録は更新し、ほかのカーネルは変更しない。
    --staging-dirはユーザーのJupyter設定を書き換えず、kernel.jsonだけを指定先へ保存する。

    Examples
    --------
    プロジェクトフォルダで、PETSc環境のPythonから呼ぶ::

        python tools/register_petsc_kernel.py
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path, default=Path(__file__).resolve().parents[2]/'env_pyaxistreamer_petsc')
    parser.add_argument('--staging-dir', type=Path)
    args = parser.parse_args()
    spec = kernel_spec(args.environment)
    if args.staging_dir:
        args.staging_dir.mkdir(parents=True, exist_ok=True)
        path = args.staging_dir/'kernel.json'
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(path)
    else:
        from jupyter_client.kernelspec import KernelSpecManager
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'kernel.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            print(KernelSpecManager().install_kernel_spec(directory, kernel_name='pyaxistreamer-petsc', user=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
