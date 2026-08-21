# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ickle_cli.py'],
    pathex=[],
    binaries=[],
    # Non-Python runtime assets the app reads by relative path at runtime
    # (web UI static files, policy TOML, JSON schema, SQL migration) --
    # without these a packaged client fails as soon as it tries to serve the
    # web UI or load policy/schema/sql files, since PyInstaller only bundles
    # importable Python code by default, not plain data directories.
    datas=[
        ('web', 'web'),
        ('config', 'config'),
        ('schemas', 'schemas'),
        ('sql', 'sql'),
    ],
    # Kept in sync with src/app.py's module_imports dict — regenerate with:
    #   python -c "import ast; d=ast.literal_eval([n.value for n in ast.walk(ast.parse(open('src/app.py').read())) if isinstance(n, ast.Assign) and getattr(n.targets[0],'id',None)=='module_imports'][0]); print(sorted(set(d.values())))"
    hiddenimports=['src.assist', 'src.autodidact', 'src.desktop_app', 'src.build_base_lm_corpus', 'src.build_clean_corpus', 'src.build_feedback_corpus', 'src.build_honest_context_package', 'src.build_preference_pairs', 'src.build_smart_corpus', 'src.chat', 'src.chat_benchmark', 'src.code_agent', 'src.code_corpus', 'src.code_evals', 'src.code_memory', 'src.continual_guard', 'src.continual_learn', 'src.dpo_train', 'src.export_onnx', 'src.federated.client', 'src.federated.flower_server', 'src.federated.inference_swarm', 'src.federated.server', 'src.federated.swarm', 'src.honesty_context_eval', 'src.hub', 'src.knowledge_modules', 'src.lora_train', 'src.mini_app', 'src.model_library', 'src.model_maintain', 'src.open_dataset_ingest', 'src.partner_loop', 'src.preflight_win11', 'src.quantize_model', 'src.reality_check', 'src.repo_index', 'src.research_memory', 'src.sanitize_training_data', 'src.self_improve', 'src.serve_control', 'src.serve_web', 'src.show_profile', 'src.skill_manager', 'src.supercharge_ickle', 'src.teacher_anthropic', 'src.teacher_ollama', 'src.teacher_opencode', 'src.teacher_registry', 'src.test_repair_loop', 'src.torickle', 'src.train', 'src.train_autopilot', 'src.train_cycle', 'src.train_intelligence_stack', 'src.trainer_orchestrator_cli', 'src.trainer_providers_cli', 'src.training_maintain', 'src.workspace_check'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'test'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ickle_client',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ickle_client',
)
