fn jacobian_color() {
    const int32_t source_idx = correspondences[corresp][1];

    JacobianType jacobian(JtJ_partial[corresp], Jtr_partial[corresp]);

    const Vector<scalar_t, 3> Tsrc_point =
	rt_cam.Transform(to_vec3<scalar_t>(src_points[source_idx]));

    scalar_t u, v;
    kcam.Project(Tsrc_point, u, v);

    scalar_t squared_residual = 0;
    const auto interp(tgt_featmap.GetInterpolator(u, v));
    const auto dx_interp(tgt_featmap.GetInterpolatorGradient(u, v, 0.005));

    for (int channel = 0; channel < tgt_featmap.channel_size; ++channel) {
      const scalar_t feat_residual =
          (src_feats[channel][source_idx] - interp.Get(channel));
      scalar_t du, dv;

      dx_interp.Get(channel, du, dv);

      scalar_t j00_proj, j02_proj, j11_proj, j12_proj;
      kcam.Dx_Project(Tsrc_point, j00_proj, j02_proj, j11_proj, j12_proj);

      const Vector<scalar_t, 3> gradk(du * j00_proj, dv * j11_proj,
                                      du * j02_proj + dv * j12_proj);

      jacobian.Compute(Tsrc_point, gradk, feat_residual);
      squared_residual += feat_residual * feat_residual;
    }

    if (squared_residual > residual_thresh * residual_thresh) {
      jacobian.Zero();
      squared_residuals[corresp] = 0.0;
    } else {
      squared_residuals[corresp] = squared_residual;
    }
  }

}
