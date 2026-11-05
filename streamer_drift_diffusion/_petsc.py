"""固定したFiPy行列に対し、PETScのKSPとLU分解を保持する限定アダプタ。"""
from weakref import finalize

from fipy.solvers.petsc import LinearLUSolver
from petsc4py import PETSc


class CachedLUSolver(LinearLUSolver):
    """FiPyのPETSc LUインターフェースで、固定行列の分解を再利用する。

    Parameters
    ----------
    tolerance : float, optional
        FiPyの相対許容値。既定の直接解の採否はPoissonSolverの独立残差で判定する。
    criterion : str, optional
        FiPy互換の基準名。PoissonSolverからはRHSを指定する。
    iterations : int, optional
        FiPy互換の反復上限。既定のpreonly/LUは1回の直接解を行う。
    precon : object, optional
        基底クラス互換の引数。基底クラスと同じく無視する。

    Notes
    -----
    Poisson専用。行列要素・形状は保持期間中に変更しない。FiPy 4.0.3の_solve_を拡張する。
    poisson_接頭辞のPETScオプションだけを読む。既定はpreonly/LU、初期推定0。
    close()またはオブジェクト回収時にKSPを破棄する。MPI領域分割への対応は別工程。

    Examples
    --------
    >>> solver = CachedLUSolver(tolerance=1e-10, criterion='RHS')
    >>> solver.close()  # 未使用でも繰り返し呼べる
    """

    def _solve_(self, L, x, b):
        """FiPyから渡された固定行列と現在の右辺でPETScを解く。

        Parameters
        ----------
        L : PETSc.Mat
            FiPyが組んだ固定行列。2回目以後も同じ行列を渡す。
        x : PETSc.Vec
            解を格納するFiPyのベクトル。
        b : PETSc.Vec
            今回の右辺。

        Returns
        -------
        PETSc.Vec
            更新したx。

        Raises
        ------
        RuntimeError
            PETScの設定・求解が失敗、または収束理由が正でない場合。

        Notes
        -----
        FiPy内部から呼ばれる。エラー時の一時ベクトル解放と電位復元はPoissonSolverが行う。
        """
        try:
            if not hasattr(self, 'ksp'):
                self.ksp = PETSc.KSP().create(comm=L.comm)
                self._release = finalize(self, self.ksp.destroy)
                self.ksp.setOptionsPrefix('poisson_')
                self.ksp.setOperators(L)
                self.ksp.setType('preonly')
                self.ksp.getPC().setType('lu')
                self.ksp.setTolerances(rtol=self.tolerance/(10*L.getSize()[0]**.5), atol=0., max_it=self.iterations)
                self.ksp.setFromOptions()
                self.ksp.setErrorIfNotConverged(True)
            self.ksp.solve(b, x)
        except PETSc.Error as exc:
            raise RuntimeError(f'PETSc Poisson solve failed: {exc}') from exc
        if self.ksp.reason <= 0:
            raise RuntimeError(f'PETSc Poisson did not converge: {self.ksp.reason}')
        self._setConvergence(suite='petsc', code=self.ksp.reason, iterations=self.ksp.its,
            residual=self.ksp.norm, ksp_solver=self.ksp.type, ksp_precon=self.ksp.getPC().type)
        return x

    def close(self):
        """保持しているKSPと、求解失敗時に残ったFiPy一時ベクトルを解放する。

        Returns
        -------
        None
            解放後も同じ固定行列で再度求解できる。KSPは次の求解時に再作成する。

        Notes
        -----
        何度呼んでもよい。行列の所有権はFiPy側にあり、ここで行列を書き換えない。

        Examples
        --------
        >>> solver = CachedLUSolver()
        >>> solver.close()
        >>> solver.close()
        """
        if hasattr(self, '_globalVectors'):
            self._cleanup()
        if hasattr(self, 'ksp'):
            self._release()
            del self.ksp
