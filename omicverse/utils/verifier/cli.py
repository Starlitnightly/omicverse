"""
CLI Tool for OmicVerse Skills Verifier - Phase 6

Command-line interface for running verification, validating descriptions,
and extracting tasks from notebooks.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional

from .skill_description_loader import SkillDescriptionLoader
from .llm_skill_selector import create_skill_selector
from .skill_description_quality import create_quality_checker
from .notebook_task_extractor import create_task_extractor
from .end_to_end_verifier import (
    create_verifier,
    VerificationRunConfig,
)


def cmd_verify(args):
    """Run end-to-end verification."""
    print("=" * 80)
    print("OmicVerse Skills Verifier - Running Verification")
    print("=" * 80)
    print()

    # Create config
    config = VerificationRunConfig(
        notebooks_dir=args.notebooks_dir,
        notebook_pattern=args.pattern,
        model=args.model,
        temperature=args.temperature,
        max_concurrent_tasks=args.max_concurrent,
        skip_notebooks=args.skip or [],
        only_categories=args.categories or None,
    )

    print(f"Notebooks directory: {config.notebooks_dir}")
    print(f"Pattern: {config.notebook_pattern}")
    print(f"Model: {config.model}")
    print(f"Max concurrent: {config.max_concurrent_tasks}")
    if config.only_categories:
        print(f"Categories filter: {', '.join(config.only_categories)}")
    print()

    # Create verifier
    print("Initializing verifier...")
    verifier = create_verifier(skills_dir=args.skills_dir)
    print(f"Loaded {len(verifier.skills)} skills")
    print()

    # Run verification
    summary = verifier.run_verification(config)

    # Generate report
    report = verifier.generate_report(summary, detailed=args.detailed)
    print(report)

    # Save report if requested
    if args.output:
        verifier.save_report(summary, args.output, detailed=args.detailed)
        print(f"\nReport saved to: {args.output}")

    # Save JSON summary if requested
    if args.json_output:
        json_data = {
            'run_id': summary.run_id,
            'timestamp': summary.timestamp,
            'total_tasks': summary.total_tasks,
            'tasks_verified': summary.tasks_verified,
            'tasks_passed': summary.tasks_passed,
            'tasks_failed': summary.tasks_failed,
            'avg_precision': summary.avg_precision,
            'avg_recall': summary.avg_recall,
            'avg_f1_score': summary.avg_f1_score,
            'avg_ordering_accuracy': summary.avg_ordering_accuracy,
            'notebooks_tested': summary.notebooks_tested,
            'skills_tested': summary.skills_tested,
            'skills_not_tested': summary.skills_not_tested,
            'category_metrics': summary.category_metrics,
            'difficulty_metrics': summary.difficulty_metrics,
            'passed_criteria': summary.passed_criteria(),
            'failed_tasks': summary.failed_tasks,
        }

        with open(args.json_output, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON summary saved to: {args.json_output}")

    # Exit with appropriate code
    if summary.passed_criteria():
        sys.exit(0)
    else:
        sys.exit(1)


def cmd_validate(args):
    """Validate skill descriptions."""
    print("=" * 80)
    print("OmicVerse Skills Verifier - Validating Skill Descriptions")
    print("=" * 80)
    print()

    # Load skills
    loader = SkillDescriptionLoader(skills_dir=args.skills_dir)
    skills = loader.load_all_descriptions()

    print(f"Loaded {len(skills)} skill descriptions")
    print()

    # Get statistics
    stats = loader.get_statistics(skills)
    print("STATISTICS")
    print("-" * 80)
    print(f"Total skills: {stats['total_skills']}")
    print(f"Average tokens: {stats['avg_token_estimate']:.1f}")
    print(f"Total tokens: {stats['total_token_estimate']}")
    print(f"Categories: {', '.join(stats['categories'])}")
    print()

    # Validate descriptions
    warnings = loader.validate_descriptions(skills)

    if not warnings:
        print("✅ All skill descriptions passed validation!")
        print()
    else:
        print("WARNINGS")
        print("-" * 80)
        issues_count = 0
        for skill_name, skill_warnings in warnings.items():
            print(f"\n{skill_name}:")
            for warning in skill_warnings:
                print(f"  ⚠️  {warning}")
                issues_count += 1

        print()
        print(f"Total issues found: {issues_count}")
        print()

    # Quality check if requested
    if args.check_quality:
        print("QUALITY METRICS")
        print("-" * 80)

        checker = create_quality_checker()
        results = checker.check_all_skills(skills)

        for skill_name, metrics in results.items():
            score_emoji = "✅" if metrics.overall_score >= 0.8 else "⚠️" if metrics.overall_score >= 0.6 else "❌"
            print(f"{score_emoji} {skill_name}: {metrics.overall_score:.2f}")
            if args.detailed:
                print(f"   Completeness: {metrics.completeness_score:.2f}")
                print(f"   Clarity: {metrics.clarity_score:.2f}")
                if metrics.recommendations:
                    print(f"   Recommendations:")
                    for rec in metrics.recommendations:
                        print(f"     - {rec}")

        summary = checker.get_quality_summary(skills)
        print()
        print(f"Average score: {summary['avg_overall_score']:.2f}")
        print(f"Skills needing improvement: {summary['skills_needing_improvement']}")

    # Exit with appropriate code
    sys.exit(0 if not warnings else 1)


def cmd_extract(args):
    """Extract tasks from notebooks."""
    print("=" * 80)
    print("OmicVerse Skills Verifier - Extracting Tasks")
    print("=" * 80)
    print()

    # Create extractor
    extractor = create_task_extractor()

    # Extract tasks
    if args.notebook:
        # Single notebook
        print(f"Extracting tasks from: {args.notebook}")
        tasks = extractor.extract_from_notebook(args.notebook)
    else:
        # Directory
        print(f"Extracting tasks from directory: {args.directory}")
        print(f"Pattern: {args.pattern}")
        tasks = extractor.extract_from_directory(args.directory, args.pattern)

    print(f"Found {len(tasks)} tasks")
    print()

    # Show tasks
    if args.detailed:
        print("TASKS")
        print("-" * 80)
        for task in tasks:
            print(f"\nTask ID: {task.task_id}")
            print(f"Notebook: {task.notebook_path}")
            print(f"Description: {task.task_description}")
            print(f"Expected skills: {', '.join(task.expected_skills)}")
            print(f"Category: {task.category}")
            print(f"Difficulty: {task.difficulty}")

    # Show coverage if skills provided
    if args.show_coverage:
        loader = SkillDescriptionLoader(skills_dir=args.skills_dir)
        skills = loader.load_all_descriptions()

        stats = extractor.get_coverage_statistics(tasks, skills)

        print("COVERAGE STATISTICS")
        print("-" * 80)
        print(f"Total tasks: {stats['total_tasks']}")
        print(f"Total notebooks: {stats['total_notebooks']}")
        print(f"Skills covered: {stats['skills_covered']}/{stats['skills_covered'] + stats['skills_not_covered']}")
        print(f"Coverage: {stats['coverage_percentage']:.1f}%")

        if stats['not_covered_skills']:
            print(f"\nSkills not covered:")
            for skill in stats['not_covered_skills']:
                print(f"  - {skill}")

    # Save to JSON if requested
    if args.output:
        extractor.save_tasks_to_json(tasks, args.output)
        print(f"\nTasks saved to: {args.output}")

    sys.exit(0)


def cmd_test_selection(args):
    """Test LLM skill selection for a task."""
    print("=" * 80)
    print("OmicVerse Skills Verifier - Testing Skill Selection")
    print("=" * 80)
    print()

    # Load skills
    loader = SkillDescriptionLoader(skills_dir=args.skills_dir)
    skills = loader.load_all_descriptions()

    print(f"Loaded {len(skills)} skills")
    print()

    # Create selector
    print(f"Creating LLM selector (model: {args.model})...")
    selector = create_skill_selector(
        skill_descriptions=skills,
        model=args.model,
        temperature=args.temperature,
    )
    print()

    # Get task description
    task_description = args.task or input("Enter task description: ")

    print(f"Task: {task_description}")
    print()
    print("Asking LLM to select skills...")
    print()

    # Select skills
    result = selector.select_skills(task_description)

    # Show results
    print("RESULTS")
    print("-" * 80)
    print(f"Selected skills: {', '.join(result.selected_skills) if result.selected_skills else 'None'}")
    print(f"Skill order: {', '.join(result.skill_order) if result.skill_order else 'N/A'}")
    print(f"\nReasoning:")
    print(result.reasoning)
    print()

    sys.exit(0)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='omicverse-verifier',
        description='OmicVerse Skills Verifier - Test and validate skill selection',
        epilog='For more information, see omicverse/utils/verifier/README.md'
    )

    # Global options
    parser.add_argument(
        '--skills-dir',
        type=str,
        help='Directory containing skill descriptions (default: .claude/skills/)',
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Verify command
    verify_parser = subparsers.add_parser(
        'verify',
        help='Run end-to-end verification'
    )
    verify_parser.add_argument(
        'notebooks_dir',
        type=str,
        help='Directory containing notebooks to verify'
    )
    verify_parser.add_argument(
        '--pattern',
        type=str,
        default='**/*.ipynb',
        help='Glob pattern for finding notebooks (default: **/*.ipynb)'
    )
    verify_parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='LLM model to use (default: gpt-4o-mini)'
    )
    verify_parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='LLM temperature (default: 0.0)'
    )
    verify_parser.add_argument(
        '--max-concurrent',
        type=int,
        default=5,
        help='Maximum concurrent tasks (default: 5)'
    )
    verify_parser.add_argument(
        '--skip',
        type=str,
        nargs='+',
        help='Notebook names to skip'
    )
    verify_parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        help='Only verify these categories (e.g., bulk single-cell)'
    )
    verify_parser.add_argument(
        '--detailed',
        action='store_true',
        help='Generate detailed report'
    )
    verify_parser.add_argument(
        '--output',
        type=str,
        help='Save report to file'
    )
    verify_parser.add_argument(
        '--json-output',
        type=str,
        help='Save JSON summary to file'
    )
    verify_parser.set_defaults(func=cmd_verify)

    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate skill descriptions'
    )
    validate_parser.add_argument(
        '--check-quality',
        action='store_true',
        help='Check quality metrics for descriptions'
    )
    validate_parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed quality metrics'
    )
    validate_parser.set_defaults(func=cmd_validate)

    # Extract command
    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract tasks from notebooks'
    )
    extract_group = extract_parser.add_mutually_exclusive_group(required=True)
    extract_group.add_argument(
        '--notebook',
        type=str,
        help='Single notebook to extract from'
    )
    extract_group.add_argument(
        '--directory',
        type=str,
        help='Directory of notebooks to extract from'
    )
    extract_parser.add_argument(
        '--pattern',
        type=str,
        default='**/*.ipynb',
        help='Glob pattern for finding notebooks (default: **/*.ipynb)'
    )
    extract_parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed task information'
    )
    extract_parser.add_argument(
        '--show-coverage',
        action='store_true',
        help='Show coverage statistics'
    )
    extract_parser.add_argument(
        '--output',
        type=str,
        help='Save tasks to JSON file'
    )
    extract_parser.set_defaults(func=cmd_extract)

    # Test selection command
    test_parser = subparsers.add_parser(
        'test-selection',
        help='Test LLM skill selection for a task'
    )
    test_parser.add_argument(
        '--task',
        type=str,
        help='Task description (will prompt if not provided)'
    )
    test_parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='LLM model to use (default: gpt-4o-mini)'
    )
    test_parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='LLM temperature (default: 0.0)'
    )
    test_parser.set_defaults(func=cmd_test_selection)

    # Parse and execute
    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        if '--debug' in sys.argv:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
