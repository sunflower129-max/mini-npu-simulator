"""표준 라이브러리만 사용하는 Mini NPU 학습용 시뮬레이터 (Python 3.8+)."""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from time import perf_counter_ns


ROOT = Path(__file__).resolve().parent
EPSILON = 1e-9
REPEATS = 1000


def normalize_label(value):
    """외부 데이터의 라벨을 Cross 또는 X로 통일한다."""
    if not isinstance(value, str):
        raise ValueError("라벨은 문자열이어야 합니다.")
    aliases = {"+": "Cross", "cross": "Cross", "x": "X"}
    label = aliases.get(value.strip().lower())
    if label is None:
        raise ValueError("알 수 없는 라벨: {!r} (Cross, +, X, x 사용)".format(value))
    return label


def validate_matrix(matrix, size, name="배열"):
    """정사각형 크기와 유한한 숫자 여부를 확인한다. bool은 숫자로 받지 않는다."""
    if not isinstance(matrix, list) or len(matrix) != size:
        raise ValueError("{}: 행 개수는 {}개여야 합니다.".format(name, size))
    for row_index, row in enumerate(matrix, 1):
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("{}: {}행의 열 개수는 {}개여야 합니다.".format(name, row_index, size))
        for value in row:
            if type(value) not in (int, float):
                raise ValueError("{}: bool/문자열 대신 숫자를 입력하세요.".format(name))
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("{}: NaN, Infinity, 지나치게 큰 숫자는 사용할 수 없습니다.".format(name))
    return matrix


def mac_2d(pattern, kernel):
    """검증된 동일 크기 행렬의 위치별 곱을 반복문으로 누적한다."""
    total = 0.0
    for row in range(len(pattern)):
        for column in range(len(pattern)):
            total += pattern[row][column] * kernel[row][column]
    return total


def flatten(matrix):
    return [value for row in matrix for value in row]


def mac_1d(pattern, kernel):
    total = 0.0
    for index in range(len(pattern)):
        total += pattern[index] * kernel[index]
    return total


def decide(score_a, score_b, label_a="Cross", label_b="X"):
    if not math.isfinite(score_a) or not math.isfinite(score_b):
        raise ValueError("MAC 결과가 유한하지 않습니다. 숫자 크기를 줄이세요.")
    if abs(score_a - score_b) < EPSILON:
        return "UNDECIDED"
    return label_a if score_a > score_b else label_b


def generate_patterns(size):
    """중앙 십자가와 두 대각선 X를 만든다. 홀수 크기만 허용한다."""
    if type(size) is not int or size < 3 or size % 2 == 0:
        raise ValueError("크기는 3 이상의 홀수여야 합니다.")
    center = size // 2
    cross = [[int(row == center or col == center) for col in range(size)] for row in range(size)]
    x_shape = [[int(row == col or row + col == size - 1) for col in range(size)] for row in range(size)]
    return cross, x_shape


def read_matrix(name, size=3):
    print("\n{}: {}줄, 각 줄에 숫자 {}개를 공백으로 구분하세요.".format(name, size, size))
    matrix = []
    while len(matrix) < size:
        raw = input("{}행: ".format(len(matrix) + 1)).strip().split()
        try:
            if len(raw) != size:
                raise ValueError("열 개수 오류")
            row = [float(token) for token in raw]
            if not all(math.isfinite(value) for value in row):
                raise ValueError("유한하지 않은 값")
        except ValueError:
            print("입력 형식 오류: 각 줄에 {}개의 유한한 숫자를 공백으로 구분해 입력하세요.".format(size))
            continue
        matrix.append(row)
    print("{} 저장 확인: {}행 × {}열 (메모리에 보관)".format(name, size, size))
    return matrix


def measure(function, *args, repeats=REPEATS):
    """I/O, 검증, 배열 변환을 제외하고 호출 구간만 반복 측정한다."""
    for _ in range(10):
        function(*args)
    elapsed_ns = 0
    for _ in range(repeats):
        start = perf_counter_ns()
        function(*args)
        elapsed_ns += perf_counter_ns() - start
    return elapsed_ns / repeats / 1_000_000


def performance(sizes, repeats):
    print("\n=== 성능 분석: 각 방식 {}회 평균 ===".format(repeats))
    print("한 필터의 MAC 호출 기준 / 생성·검증·평탄화·출력 시간 제외")
    print("크기       2차원 평균(ms)   1차원 평균(ms)   MAC 횟수(N²)   2D/1D")
    rows = []
    for size in sizes:
        pattern, _ = generate_patterns(size)
        kernel = [row[:] for row in pattern]
        flat_pattern, flat_kernel = flatten(pattern), flatten(kernel)
        score_2d = mac_2d(pattern, kernel)
        score_1d = mac_1d(flat_pattern, flat_kernel)
        if score_2d != score_1d:
            raise ValueError("2차원과 1차원 MAC 결과가 다릅니다.")
        time_2d = measure(mac_2d, pattern, kernel, repeats=repeats)
        time_1d = measure(mac_1d, flat_pattern, flat_kernel, repeats=repeats)
        ratio = time_2d / time_1d if time_1d else 0.0
        print("{:<9} {:>14.6f} {:>16.6f} {:>14} {:>8.2f}".format(
            "{}x{}".format(size, size), time_2d, time_1d, size * size, ratio))
        rows.append((size, time_2d, time_1d, ratio))
    print("MAC 1회 = 곱셈 1회 + 누적 덧셈 1회. 두 필터 판정은 2N² MAC입니다.")
    print("1차원 방식이 항상 빠른 것은 아닙니다. 장비와 측정 시점에 따라 달라집니다.")
    return rows


def manual_mode(repeats):
    kernel_a = read_matrix("필터 A")
    kernel_b = read_matrix("필터 B")
    pattern = read_matrix("입력 패턴")
    start = perf_counter_ns()
    score_a = mac_2d(pattern, kernel_a)
    score_b = mac_2d(pattern, kernel_b)
    elapsed = (perf_counter_ns() - start) / 1_000_000
    result = decide(score_a, score_b, "A", "B")
    print("\n=== MAC 결과 ===")
    print("A 점수: {:.17g}".format(score_a))
    print("B 점수: {:.17g}".format(score_b))
    print("두 필터 연산 시간: {:.6f} ms".format(elapsed))
    print("판정: {}".format("판정 불가 (UNDECIDED)" if result == "UNDECIDED" else result))
    performance([3], repeats)


def load_filters(raw_filters):
    valid, errors = {}, {}
    if not isinstance(raw_filters, dict):
        return valid, {"전체": "filters는 객체여야 합니다."}
    for size in (5, 13, 25):
        key = "size_{}".format(size)
        try:
            group = raw_filters.get(key)
            if not isinstance(group, dict):
                raise ValueError("필터 그룹이 없거나 객체가 아닙니다.")
            normalized = {}
            for label, matrix in group.items():
                standard = normalize_label(label)
                if standard in normalized:
                    raise ValueError("정규화 후 중복된 필터 라벨: " + standard)
                normalized[standard] = validate_matrix(matrix, size, key + "/" + label)
            if set(normalized) != {"Cross", "X"}:
                raise ValueError("Cross와 X 필터가 모두 필요합니다.")
            valid[size] = normalized
            print("{} 필터 로드 완료 (Cross, X)".format(key))
        except (ValueError, OverflowError) as error:
            errors[key] = str(error)
            print("{} 필터 오류: {}".format(key, error))
    return valid, errors


def analyze_document(document):
    """파일 I/O와 분리한 분석 함수. 손상된 케이스도 결과에 FAIL로 남긴다."""
    if not isinstance(document, dict):
        raise ValueError("JSON 최상위는 객체여야 합니다.")
    if not isinstance(document.get("patterns"), dict) or not document["patterns"]:
        raise ValueError("patterns는 비어 있지 않은 객체여야 합니다.")
    print("\n=== 필터 로드 ===")
    filters, filter_errors = load_filters(document.get("filters"))
    results = []
    print("\n=== 패턴 분석 ===")
    for identifier, case in document["patterns"].items():
        record = {"id": identifier, "passed": False, "reason": ""}
        print("\n--- {} ---".format(identifier))
        try:
            match = re.fullmatch(r"size_(5|13|25)_([0-9]+)", identifier)
            if match is None:
                raise ValueError("키 형식 오류: size_5_1 같은 size_{N}_{idx} 형식이 필요합니다.")
            size = int(match.group(1))
            if size not in filters:
                reason = filter_errors.get("size_{}".format(size), filter_errors.get("전체", "필터 없음"))
                raise ValueError("사용할 필터 오류: " + reason)
            if not isinstance(case, dict):
                raise ValueError("케이스는 input과 expected를 가진 객체여야 합니다.")
            pattern = validate_matrix(case["input"], size, "input")
            expected = normalize_label(case["expected"])
            score_cross = mac_2d(pattern, filters[size]["Cross"])
            score_x = mac_2d(pattern, filters[size]["X"])
            result = decide(score_cross, score_x)
            record.update(result=result, expected=expected, score_cross=score_cross, score_x=score_x)
            record["passed"] = result == expected
            if result == "UNDECIDED":
                record["reason"] = "허용오차 내 동점: UNDECIDED는 Cross/X 정답과 일치하지 않음"
            elif result != expected:
                record["reason"] = "판정 {}와 expected {} 불일치: 입력·필터·라벨 확인 필요".format(result, expected)
            print("Cross 점수: {:.17g} | X 점수: {:.17g}".format(score_cross, score_x))
            print("판정: {} | expected: {} | {}".format(result, expected, "PASS" if record["passed"] else "FAIL"))
        except (KeyError, ValueError, OverflowError) as error:
            record["reason"] = "데이터/스키마 또는 수치 오류: " + str(error)
            print("FAIL | " + record["reason"])
        results.append(record)
    return results, filter_errors


def json_mode(path, repeats):
    try:
        with path.open("r", encoding="utf-8") as file:
            document = json.load(file)
        results, filter_errors = analyze_document(document)
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        print("data.json 분석을 시작할 수 없습니다: {}".format(error))
        return 1
    performance([3, 5, 13, 25], repeats)
    failed = [record for record in results if not record["passed"]]
    print("\n=== 결과 요약 ===")
    print("총 테스트: {}개 | 통과: {}개 | 실패: {}개".format(len(results), len(results) - len(failed), len(failed)))
    for record in failed:
        print("- {}: {}".format(record["id"], record["reason"]))
    if filter_errors:
        print("필터 로드 오류 그룹: {}개 (케이스 수와 별도 집계)".format(len(filter_errors)))
    return 1 if failed or filter_errors else 0


def demo_document():
    filters, patterns = {}, {}
    for size in (5, 13, 25):
        cross, x_shape = generate_patterns(size)
        filters["size_{}".format(size)] = {"cross": cross, "x": x_shape}
        patterns["size_{}_1".format(size)] = {"input": cross, "expected": "+"}
        patterns["size_{}_2".format(size)] = {"input": x_shape, "expected": "x"}
        weighted = [[value * 0.1 for value in row] for row in cross]
        patterns["size_{}_3".format(size)] = {"input": weighted, "expected": "Cross"}
    patterns["size_13_4"] = {"input": [[0.0] * 13 for _ in range(13)], "expected": "x"}
    patterns["size_5_4"] = {"input": [[1, 0], [0, 1]], "expected": "+"}
    cross, _ = generate_patterns(25)
    patterns["size_25_4"] = {"input": cross, "expected": "triangle"}
    return {"filters": filters, "patterns": patterns}


def main():
    parser = argparse.ArgumentParser(description="Mini NPU Simulator")
    parser.add_argument("--mode", choices=("manual", "json"))
    parser.add_argument("--data", type=Path, default=ROOT / "data.json")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--generate-demo", action="store_true", help="예제 JSON 생성 (기존 파일 덮어쓰기 금지)")
    parser.add_argument("--generate", type=int, metavar="N", help="홀수 N 크기의 Cross/X 출력")
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error("반복 횟수는 10 이상이어야 합니다.")
    try:
        if args.generate_demo:
            with args.data.open("x", encoding="utf-8") as file:
                json.dump(demo_document(), file, ensure_ascii=False, indent=2)
            print("예제 데이터를 생성했습니다: {}".format(args.data))
            return 0
        if args.generate is not None:
            cross, x_shape = generate_patterns(args.generate)
            for label, matrix in (("Cross", cross), ("X", x_shape)):
                print(label)
                for row in matrix:
                    print(" ".join(map(str, row)))
            return 0
        print("=== Mini NPU Simulator ===")
        mode = args.mode
        while mode is None:
            print("1. 사용자 입력 (3x3)\n2. data.json 분석\n0. 종료")
            choice = input("선택: ").strip()
            if choice == "0":
                return 0
            mode = {"1": "manual", "2": "json"}.get(choice)
            if mode is None:
                print("0, 1, 2 중 하나를 입력하세요.")
        if mode == "manual":
            manual_mode(args.repeats)
            return 0
        return json_mode(args.data, args.repeats)
    except (KeyboardInterrupt, EOFError):
        print("\n입력이 중단되어 안전하게 종료합니다.")
        return 0
    except (OSError, ValueError, OverflowError) as error:
        print("처리할 수 없습니다: {}".format(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
